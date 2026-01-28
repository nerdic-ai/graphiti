import asyncio
import contextlib
import logging
from dataclasses import dataclass

from fastapi import APIRouter, status
from graphiti_core.nodes import EpisodeType  # type: ignore
from graphiti_core.utils.maintenance.graph_data_operations import clear_data  # type: ignore

from graph_service.config import get_settings
from graph_service.dto import AddEntityNodeRequest, AddMessagesRequest, Message, Result
from graph_service.zep_graphiti import ZepGraphiti, ZepGraphitiDep, create_graph_driver

logger = logging.getLogger(__name__)


@dataclass
class MessageJob:
    """Data class to hold message processing job details."""

    message: Message
    group_id: str


class AsyncWorker:
    def __init__(self):
        self.queue: asyncio.Queue[MessageJob] = asyncio.Queue()
        self.task = None
        self._graphiti: ZepGraphiti | None = None

    async def start(self):
        # Create a persistent graphiti client for the worker
        settings = get_settings()
        driver = create_graph_driver(settings)
        self._graphiti = ZepGraphiti(graph_driver=driver)

        # Configure LLM client
        if settings.openai_base_url is not None:
            self._graphiti.llm_client.config.base_url = settings.openai_base_url
        if settings.openai_api_key is not None:
            self._graphiti.llm_client.config.api_key = settings.openai_api_key
        if settings.model_name is not None:
            self._graphiti.llm_client.model = settings.model_name

        self.task = asyncio.create_task(self.worker())

    async def stop(self):
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
        while not self.queue.empty():
            self.queue.get_nowait()
        if self._graphiti:
            await self._graphiti.close()
            self._graphiti = None

    async def worker(self):
        while True:
            try:
                job = await self.queue.get()
                logger.info(f'Processing job (queue size: {self.queue.qsize()})')
                print(f'Processing job (queue size: {self.queue.qsize()})')
                await self._process_message(job)
                logger.info('Job completed successfully')
                print('Job completed successfully')
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f'Error processing job: {e}')
                print(f'Error processing job: {e}')

    async def _process_message(self, job: MessageJob):
        if not self._graphiti:
            raise RuntimeError('Graphiti client not initialized')

        m = job.message
        # Note: Don't pass uuid to add_episode for new episodes.
        # The uuid parameter is for retrieving/updating EXISTING episodes.
        # Graphiti will auto-generate a UUID for new episodes.
        await self._graphiti.add_episode(
            group_id=job.group_id,
            name=m.name or m.uuid or '',
            episode_body=f'{m.role or ""}({m.role_type}): {m.content}',
            reference_time=m.timestamp,
            source=EpisodeType.message,
            source_description=m.source_description,
        )


async_worker = AsyncWorker()


router = APIRouter()


@router.post('/messages', status_code=status.HTTP_202_ACCEPTED)
async def add_messages(
    request: AddMessagesRequest,
    graphiti: ZepGraphitiDep,  # Keep for API consistency, but not used for async processing
):
    for m in request.messages:
        job = MessageJob(message=m, group_id=request.group_id)
        await async_worker.queue.put(job)

    return Result(message='Messages added to processing queue', success=True)


@router.post('/entity-node', status_code=status.HTTP_201_CREATED)
async def add_entity_node(
    request: AddEntityNodeRequest,
    graphiti: ZepGraphitiDep,
):
    node = await graphiti.save_entity_node(
        uuid=request.uuid,
        group_id=request.group_id,
        name=request.name,
        summary=request.summary,
    )
    return node


@router.delete('/entity-edge/{uuid}', status_code=status.HTTP_200_OK)
async def delete_entity_edge(uuid: str, graphiti: ZepGraphitiDep):
    await graphiti.delete_entity_edge(uuid)
    return Result(message='Entity Edge deleted', success=True)


@router.delete('/group/{group_id}', status_code=status.HTTP_200_OK)
async def delete_group(group_id: str, graphiti: ZepGraphitiDep):
    await graphiti.delete_group(group_id)
    return Result(message='Group deleted', success=True)


@router.delete('/episode/{uuid}', status_code=status.HTTP_200_OK)
async def delete_episode(uuid: str, graphiti: ZepGraphitiDep):
    await graphiti.delete_episodic_node(uuid)
    return Result(message='Episode deleted', success=True)


@router.post('/clear', status_code=status.HTTP_200_OK)
async def clear(
    graphiti: ZepGraphitiDep,
):
    await clear_data(graphiti.driver)
    await graphiti.build_indices_and_constraints()
    return Result(message='Graph cleared', success=True)
