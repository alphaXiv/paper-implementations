import asyncio
import datetime
import unittest
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError

# Import the classes we're testing
from olmocr.work_queue import S3Backend, WorkItem, WorkQueue


class TestS3WorkQueue(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.s3_client = Mock()
        self.s3_client.exceptions.ClientError = ClientError
        self.backend = S3Backend(self.s3_client, "s3://test-bucket/workspace")
        self.work_queue = WorkQueue(self.backend)
        self.sample_paths = [
            "s3://test-bucket/data/file1.pdf",
            "s3://test-bucket/data/file2.pdf",
            "s3://test-bucket/data/file3.pdf",
        ]

    def tearDown(self):
        """Clean up after each test method."""
        pass

    def test_compute_workgroup_hash(self):
        """Test hash computation is deterministic and correct"""
        paths = [
            "s3://test-bucket/data/file2.pdf",
            "s3://test-bucket/data/file1.pdf",
        ]

        # Hash should be the same regardless of order
        hash1 = WorkQueue._compute_workgroup_hash(paths)
        hash2 = WorkQueue._compute_workgroup_hash(reversed(paths))
        self.assertEqual(hash1, hash2)

    def test_init(self):
        """Test initialization of S3Backend"""
        client = Mock()
        backend = S3Backend(client, "s3://test-bucket/workspace/")

        self.assertEqual(backend.workspace_path, "s3://test-bucket/workspace")
        self.assertEqual(backend._index_path, "s3://test-bucket/workspace/work_index_list.csv.zstd")
        self.assertEqual(backend._output_glob, "s3://test-bucket/workspace/done_flags/*.flag")

    def asyncSetUp(self):
        """Set up async test fixtures"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def asyncTearDown(self):
        """Clean up async test fixtures"""
        self.loop.close()

    def async_test(f):
        """Decorator for async test methods"""

        def wrapper(*args, **kwargs):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(f(*args, **kwargs))
            finally:
                loop.close()

        return wrapper

    @async_test
    async def test_populate_queue_new_items(self):
        """Test populating queue with new items"""
        # Mock empty existing index
        with patch("olmocr.work_queue.download_zstd_csv", return_value=[]):
            with patch("olmocr.work_queue.upload_zstd_csv") as mock_upload:
                await self.work_queue.populate_queue(self.sample_paths, items_per_group=2)

                # Verify upload was called with correct data
                self.assertEqual(mock_upload.call_count, 1)
                _, _, lines = mock_upload.call_args[0]

                # Should create 2 work groups (2 files + 1 file)
                self.assertEqual(len(lines), 2)

                # Verify format of uploaded lines
                for line in lines:
                    parts = WorkQueue._decode_csv_row(line)
                    self.assertGreaterEqual(len(parts), 2)  # Hash + at least one path
                    self.assertEqual(len(parts[0]), 40)  # SHA1 hash length

    @async_test
    async def test_populate_queue_existing_items(self):
        """Test populating queue with mix of new and existing items"""
        existing_paths = ["s3://test-bucket/data/existing1.pdf"]
        new_paths = ["s3://test-bucket/data/new1.pdf"]

        # Create existing index content
        existing_hash = WorkQueue._compute_workgroup_hash(existing_paths)
        existing_line = WorkQueue._encode_csv_row([existing_hash] + existing_paths)

        with patch("olmocr.work_queue.download_zstd_csv", return_value=[existing_line]):
            with patch("olmocr.work_queue.upload_zstd_csv") as mock_upload:
                await self.work_queue.populate_queue(existing_paths + new_paths, items_per_group=1)

                # Verify upload called with both existing and new items
                _, _, lines = mock_upload.call_args[0]
                self.assertEqual(len(lines), 2)
                self.assertIn(existing_line, lines)

    @async_test
    async def test_initialize_queue(self):
        """Test queue initialization"""
        # Mock work items and completed items
        work_paths = ["s3://test/file1.pdf", "s3://test/file2.pdf"]
        work_hash = WorkQueue._compute_workgroup_hash(work_paths)
        work_line = WorkQueue._encode_csv_row([work_hash] + work_paths)

        completed_items = [f"s3://test-bucket/workspace/done_flags/done_{work_hash}.flag"]

        with patch("olmocr.work_queue.download_zstd_csv", return_value=[work_line]):
            with patch("olmocr.work_queue.expand_s3_glob", return_value=completed_items):
                count = await self.work_queue.initialize_queue()

                # Queue should be empty since all work is completed
                self.assertEqual(count, 0)

    @async_test
    async def test_is_completed(self):
        """Test completed work check"""
        work_hash = "testhash123"

        # Test completed work
        self.s3_client.head_object.return_value = {"LastModified": datetime.datetime.now(datetime.timezone.utc)}
        self.assertTrue(await self.backend.is_completed(work_hash))

        # Test incomplete work
        self.s3_client.head_object.side_effect = ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        self.assertFalse(await self.backend.is_completed(work_hash))

    @async_test
    async def test_get_work(self):
        """Test getting work items"""
        # Setup test data
        work_item = WorkItem(hash="testhash123", work_paths=["s3://test/file1.pdf"])
        await self.work_queue._queue.put(work_item)

        # Test getting available work
        self.s3_client.head_object.side_effect = ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        result = await self.work_queue.get_work()
        self.assertEqual(result, work_item)

        # Verify lock file was created
        self.s3_client.put_object.assert_called_once()
        key = self.s3_client.put_object.call_args[1]["Key"]
        self.assertTrue(key.endswith(f"worker_{work_item.hash}.lock"))

    @async_test
    async def test_get_work_completed(self):
        """Test getting work that's already completed"""
        work_item = WorkItem(hash="testhash123", work_paths=["s3://test/file1.pdf"])
        await self.work_queue._queue.put(work_item)

        # Simulate completed work
        self.s3_client.head_object.return_value = {"LastModified": datetime.datetime.now(datetime.timezone.utc)}

        result = await self.work_queue.get_work()
        self.assertIsNone(result)  # Should skip completed work

    @async_test
    async def test_get_work_locked(self):
        """Test getting work that's locked by another worker"""
        work_item = WorkItem(hash="testhash123", work_paths=["s3://test/file1.pdf"])
        await self.work_queue._queue.put(work_item)

        # Simulate active lock
        recent_time = datetime.datetime.now(datetime.timezone.utc)
        self.s3_client.head_object.side_effect = [
            ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"),  # Not completed
            {"LastModified": recent_time},  # Active lock
        ]

        result = await self.work_queue.get_work()
        self.assertIsNone(result)  # Should skip locked work

    @async_test
    async def test_get_work_stale_lock(self):
        """Test getting work with a stale lock"""
        work_item = WorkItem(hash="testhash123", work_paths=["s3://test/file1.pdf"])
        await self.work_queue._queue.put(work_item)

        # Simulate stale lock
        stale_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)
        self.s3_client.head_object.side_effect = [
            ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"),  # Not completed
            {"LastModified": stale_time},  # Stale lock
        ]

        result = await self.work_queue.get_work()
        self.assertEqual(result, work_item)  # Should take work with stale lock

    @async_test
    async def test_mark_done(self):
        """Test marking work as done"""
        work_item = WorkItem(hash="testhash123", work_paths=["s3://test/file1.pdf"])
        await self.work_queue._queue.put(work_item)

        await self.work_queue.mark_done(work_item)

        # Verify done flag was created and lock file was deleted
        # Check put_object was called for done flag
        put_calls = self.s3_client.put_object.call_args_list
        self.assertEqual(len(put_calls), 1)
        done_flag_key = put_calls[0][1]["Key"]
        self.assertTrue(done_flag_key.endswith(f"done_{work_item.hash}.flag"))

        # Verify lock file was deleted
        self.s3_client.delete_object.assert_called_once()
        key = self.s3_client.delete_object.call_args[1]["Key"]
        self.assertTrue(key.endswith(f"worker_{work_item.hash}.lock"))

    @async_test
    async def test_paths_with_commas(self):
        """Test handling of paths that contain commas"""
        # Create paths with commas in them
        paths_with_commas = ["s3://test-bucket/data/file1,with,commas.pdf", "s3://test-bucket/data/file2,comma.pdf", "s3://test-bucket/data/file3.pdf"]

        # Mock empty existing index for initial population
        with patch("olmocr.work_queue.download_zstd_csv", return_value=[]):
            with patch("olmocr.work_queue.upload_zstd_csv") as mock_upload:
                # Populate the queue with these paths
                await self.work_queue.populate_queue(paths_with_commas, items_per_group=3)

                # Capture what would be written to the index
                _, _, lines = mock_upload.call_args[0]

                # Now simulate reading back these lines (which have commas in the paths)
                with patch("olmocr.work_queue.download_zstd_csv", return_value=lines):
                    with patch("olmocr.work_queue.expand_s3_glob", return_value=[]):
                        # Initialize a fresh queue from these lines
                        await self.work_queue.initialize_queue()

                        # Mock ClientError for head_object (file doesn't exist) - need to handle multiple calls
                        self.s3_client.head_object.side_effect = [
                            ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"),  # done flag check
                            ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"),  # worker lock check
                        ]

                        # Get a work item
                        work_item = await self.work_queue.get_work()

                        # Now verify we get a work item
                        self.assertIsNotNone(work_item, "Should get a work item")

                        # Verify the work item has the correct number of paths
                        self.assertEqual(len(work_item.work_paths), len(paths_with_commas), "Work item should have the correct number of paths")

                        # Check that all original paths with commas are preserved
                        for path in paths_with_commas:
                            print(path)
                            self.assertIn(path, work_item.work_paths, f"Path with commas should be preserved: {path}")

    def test_queue_size(self):
        """Test queue size property"""
        self.assertEqual(self.work_queue.size, 0)

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.loop.run_until_complete(self.work_queue._queue.put(WorkItem(hash="test1", work_paths=["path1"])))
        self.assertEqual(self.work_queue.size, 1)

        self.loop.run_until_complete(self.work_queue._queue.put(WorkItem(hash="test2", work_paths=["path2"])))
        self.assertEqual(self.work_queue.size, 2)

        self.loop.close()


if __name__ == "__main__":
    unittest.main()
