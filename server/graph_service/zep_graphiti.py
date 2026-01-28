import logging
from datetime import datetime
from typing import Annotated

from fastapi import Depends, HTTPException
from graphiti_core import Graphiti  # type: ignore
from graphiti_core.driver.driver import GraphDriver, GraphProvider  # type: ignore
from graphiti_core.edges import EntityEdge  # type: ignore
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.llm_client import LLMClient  # type: ignore
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode  # type: ignore

from graph_service.config import Settings, ZepEnvDep
from graph_service.dto import FactResult

logger = logging.getLogger(__name__)


def create_graph_driver(settings: Settings) -> GraphDriver:
    """Create the appropriate graph driver based on DATABASE_PROVIDER setting."""
    if settings.database_provider == 'falkordb':
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        # Prefer public host for external connections (local dev), fall back to internal
        host = settings.falkor_public_host or settings.falkor_host
        port = settings.falkor_public_port or settings.falkor_port

        if not host:
            raise ValueError('FalkorDB host not configured. Set FALKOR_HOST or FALKOR_PUBLIC_HOST')

        return FalkorDriver(
            host=host,
            port=port,
            username=settings.falkor_username,
            password=settings.falkor_password,
            database=settings.falkor_database,
        )
    else:
        # Neo4j (default)
        from graphiti_core.driver.neo4j_driver import Neo4jDriver

        if not settings.neo4j_uri or not settings.neo4j_user or not settings.neo4j_password:
            raise ValueError('Neo4j credentials not fully configured')

        return Neo4jDriver(
            settings.neo4j_uri,
            settings.neo4j_user,
            settings.neo4j_password,
        )


class ZepGraphiti(Graphiti):
    def __init__(self, graph_driver: GraphDriver, llm_client: LLMClient | None = None):
        super().__init__(graph_driver=graph_driver, llm_client=llm_client)

    def _get_driver_for_group(self, group_id: str) -> GraphDriver:
        """Get a driver configured for the specified group_id.

        For FalkorDB, this clones the driver to use group_id as the database name.
        For Neo4j, returns the same driver (Neo4j uses group_id as a property filter).
        """
        if (
            self.driver.provider == GraphProvider.FALKORDB
            and group_id != self.driver._database
        ):
            return self.driver.clone(database=group_id)
        return self.driver

    async def retrieve_episodes(
        self,
        reference_time: datetime,
        last_n: int = 10,
        group_ids: list[str] | None = None,
        source: EpisodeType | None = None,
        driver: GraphDriver | None = None,
        saga: str | None = None,
    ) -> list[EpisodicNode]:
        """Override to handle FalkorDB's multi-database model.

        FalkorDB stores each group_id in a separate database/graph, so we need
        to query the correct database for each group_id.
        """
        if driver is None:
            driver = self.driver

        # For FalkorDB with specific group_ids, query each group's database
        if (
            driver.provider == GraphProvider.FALKORDB
            and group_ids
            and len(group_ids) > 0
        ):
            all_episodes: list[EpisodicNode] = []
            for group_id in group_ids:
                group_driver = self._get_driver_for_group(group_id)
                # Query with group_ids=None since we're already in the right database
                episodes = await super().retrieve_episodes(
                    reference_time=reference_time,
                    last_n=last_n,
                    group_ids=None,  # Don't filter by group_id, we're in the right DB
                    source=source,
                    driver=group_driver,
                    saga=saga,
                )
                all_episodes.extend(episodes)

            # Sort by valid_at and limit to last_n
            all_episodes.sort(key=lambda e: e.valid_at)
            return all_episodes[-last_n:] if len(all_episodes) > last_n else all_episodes

        # For Neo4j or no specific group_ids, use default behavior
        return await super().retrieve_episodes(
            reference_time=reference_time,
            last_n=last_n,
            group_ids=group_ids,
            source=source,
            driver=driver,
            saga=saga,
        )

    async def search(
        self,
        query: str,
        center_node_uuid: str | None = None,
        group_ids: list[str] | None = None,
        num_results: int = 10,
        search_filter=None,
        driver: GraphDriver | None = None,
    ) -> list[EntityEdge]:
        """Override to handle FalkorDB's multi-database model.

        FalkorDB stores each group_id in a separate database/graph, so we need
        to search the correct database for each group_id.
        """
        # For FalkorDB with specific group_ids, search each group's database
        if (
            self.driver.provider == GraphProvider.FALKORDB
            and group_ids
            and len(group_ids) > 0
        ):
            all_edges: list[EntityEdge] = []
            for group_id in group_ids:
                # Clone driver for this group_id
                group_driver = self._get_driver_for_group(group_id)
                # Temporarily swap driver
                original_driver = self.driver
                self.driver = group_driver
                self.clients.driver = group_driver
                try:
                    # Search with group_ids=None since we're in the right database
                    edges = await super().search(
                        query=query,
                        center_node_uuid=center_node_uuid,
                        group_ids=None,  # Don't filter by group_id
                        num_results=num_results,
                        search_filter=search_filter,
                        driver=group_driver,
                    )
                    all_edges.extend(edges)
                finally:
                    # Restore original driver
                    self.driver = original_driver
                    self.clients.driver = original_driver

            # Return top num_results (edges already have scores for ranking)
            return all_edges[:num_results]

        # For Neo4j or no specific group_ids, use default behavior
        return await super().search(
            query=query,
            center_node_uuid=center_node_uuid,
            group_ids=group_ids,
            num_results=num_results,
            search_filter=search_filter,
            driver=driver,
        )

    async def save_entity_node(self, name: str, uuid: str, group_id: str, summary: str = ''):
        new_node = EntityNode(
            name=name,
            uuid=uuid,
            group_id=group_id,
            summary=summary,
        )
        # Use the driver for the target group_id
        driver = self._get_driver_for_group(group_id)
        await new_node.generate_name_embedding(self.embedder)
        await new_node.save(driver)
        return new_node

    async def get_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            return edge
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_group(self, group_id: str):
        # Use the driver for the target group_id
        driver = self._get_driver_for_group(group_id)

        try:
            edges = await EntityEdge.get_by_group_ids(driver, [group_id])
        except GroupsEdgesNotFoundError:
            logger.warning(f'No edges found for group {group_id}')
            edges = []

        nodes = await EntityNode.get_by_group_ids(driver, [group_id])

        episodes = await EpisodicNode.get_by_group_ids(driver, [group_id])

        for edge in edges:
            await edge.delete(driver)

        for node in nodes:
            await node.delete(driver)

        for episode in episodes:
            await episode.delete(driver)

    async def delete_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            await edge.delete(self.driver)
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_episodic_node(self, uuid: str):
        try:
            episode = await EpisodicNode.get_by_uuid(self.driver, uuid)
            await episode.delete(self.driver)
        except NodeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e


async def get_graphiti(settings: ZepEnvDep):
    driver = create_graph_driver(settings)
    client = ZepGraphiti(graph_driver=driver)

    if settings.openai_base_url is not None:
        client.llm_client.config.base_url = settings.openai_base_url
    if settings.openai_api_key is not None:
        client.llm_client.config.api_key = settings.openai_api_key
    if settings.model_name is not None:
        client.llm_client.model = settings.model_name

    try:
        yield client
    finally:
        await client.close()


async def initialize_graphiti(settings: Settings):
    driver = create_graph_driver(settings)
    client = ZepGraphiti(graph_driver=driver)
    await client.build_indices_and_constraints()
    await client.close()


def get_fact_result_from_edge(edge: EntityEdge):
    return FactResult(
        uuid=edge.uuid,
        name=edge.name,
        fact=edge.fact,
        valid_at=edge.valid_at,
        invalid_at=edge.invalid_at,
        created_at=edge.created_at,
        expired_at=edge.expired_at,
    )


ZepGraphitiDep = Annotated[ZepGraphiti, Depends(get_graphiti)]
