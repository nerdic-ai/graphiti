import logging
from typing import Annotated

from fastapi import Depends, HTTPException
from graphiti_core import Graphiti  # type: ignore
from graphiti_core.driver.driver import GraphDriver  # type: ignore
from graphiti_core.edges import EntityEdge  # type: ignore
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.llm_client import LLMClient  # type: ignore
from graphiti_core.nodes import EntityNode, EpisodicNode  # type: ignore

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

    async def save_entity_node(self, name: str, uuid: str, group_id: str, summary: str = ''):
        new_node = EntityNode(
            name=name,
            uuid=uuid,
            group_id=group_id,
            summary=summary,
        )
        await new_node.generate_name_embedding(self.embedder)
        await new_node.save(self.driver)
        return new_node

    async def get_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            return edge
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_group(self, group_id: str):
        try:
            edges = await EntityEdge.get_by_group_ids(self.driver, [group_id])
        except GroupsEdgesNotFoundError:
            logger.warning(f'No edges found for group {group_id}')
            edges = []

        nodes = await EntityNode.get_by_group_ids(self.driver, [group_id])

        episodes = await EpisodicNode.get_by_group_ids(self.driver, [group_id])

        for edge in edges:
            await edge.delete(self.driver)

        for node in nodes:
            await node.delete(self.driver)

        for episode in episodes:
            await episode.delete(self.driver)

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
