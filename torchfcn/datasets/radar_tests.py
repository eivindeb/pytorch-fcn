import unittest
from pathlib import Path

class RadarDataLoaderTests(unittest.TestCase):
    def setUp(self):
        from radar import RadarDatasetFolder
        root = Path("/home/eivind/Documents/polarlys_datasets")
        cfg = root / "test" / "polarlys_cfg_test.txt"
        self.dataset = RadarDatasetFolder(str(root), cfg=str(cfg), split="train", dataset_name="test")

    def test_ais_targets_to_list(self):
        file = self.dataset.files[self.dataset.split][0]["data"][0]
        basename = Path(file).stem
        rel_path = Path(file).relative_to(self.dataset.data_folder)
        t = self.dataset.data_loader.get_time_from_basename(basename)
        sensor, sensor_index = self.dataset.data_loader.get_sensor_from_basename(str(rel_path))

        ais_targets = self.dataset.data_loader.load_ais_targets_sensor(t, sensor, sensor_index)
        ais_targets_list = self.dataset.ais_targets_to_list(ais_targets)
        self.assertTrue(len(ais_targets) == len(ais_targets_list))
        self.assertTrue(type(ais_targets_list[0]) == list)
        self.assertTrue(type(ais_targets_list[0][0]) == int)

    def test_ais_targets_to_string(self):
        file = self.dataset.files[self.dataset.split][0]["data"][0]
        basename = Path(file).stem
        rel_path = Path(file).relative_to(self.dataset.data_folder)
        t = self.dataset.data_loader.get_time_from_basename(basename)
        sensor, sensor_index = self.dataset.data_loader.get_sensor_from_basename(str(rel_path))

        ais_targets = self.dataset.data_loader.load_ais_targets_sensor(t, sensor, sensor_index)
        ais_targets_string = self.dataset.ais_targets_to_string(ais_targets)
        self.assertTrue(type(ais_targets_string) == str)
        self.assertTrue(len(ais_targets_string.split("/")) == len(ais_targets))
        self.assertTrue(ais_targets_string != "[]")

        ais_targets_string = self.dataset.ais_targets_to_string([])
        self.assertTrue(type(ais_targets_string) == str)
        self.assertEqual(ais_targets_string, "[]")

    def test_ais_targets_string_to_list(self):
        ais_targets_string = "[300, 50]/[400, 20]/[200, 503]"
        ais_targets = self.dataset.ais_targets_string_to_list(ais_targets_string)
        self.assertTrue(type(ais_targets) == list)
        self.assertTrue(type(ais_targets[0]) == list)
        self.assertTrue(len(ais_targets) == len(ais_targets_string.split("/")))

        ais_targets = self.dataset.ais_targets_string_to_list("")
        self.assertEqual(ais_targets, None)

        ais_targets = self.dataset.ais_targets_string_to_list("[]")
        self.assertEqual(ais_targets, [])

    def test_set_data_ranges(self):
        self.assertEqual(len(self.dataset.data_ranges), 3)  # default

        with self.assertRaises(ValueError):
            self.dataset.set_data_ranges(-1, 0)

        max_h = self.dataset.image_height
        max_w = self.dataset.image_width

        self.dataset.set_data_ranges(0, 0)
        self.assertEqual(len(self.dataset.data_ranges), 1)
        self.assertEqual(self.dataset.data_ranges[0], (slice(0, max_h, None), slice(0, max_w, None)))

        self.dataset.set_data_ranges(0, 1)
        self.assertEqual(len(self.dataset.data_ranges), 2)
        self.assertEqual(self.dataset.data_ranges[0], (slice(0, max_h, None), slice(0, int(max_w / 2), None)))
        self.assertEqual(self.dataset.data_ranges[1], (slice(0, max_h, None), slice(int(max_w / 2), max_w, None)))

        self.dataset.set_data_ranges(1, 0)
        self.assertEqual(len(self.dataset.data_ranges), 2)
        self.assertEqual(self.dataset.data_ranges[0], (slice(0, int(max_h / 2), None), slice(0, max_w, None)))
        self.assertEqual(self.dataset.data_ranges[1], (slice(int(max_h / 2), max_h, None), slice(0, max_w, None)))

        self.dataset.set_data_ranges(1, 1)
        self.assertEqual(len(self.dataset.data_ranges), 4)
        self.assertEqual(self.dataset.data_ranges[0], (slice(0, int(max_h / 2), None), slice(0, int(max_w / 2), None)))
        self.assertEqual(self.dataset.data_ranges[1], (slice(0, int(max_h / 2), None), slice(int(max_w / 2), max_w, None)))
        self.assertEqual(self.dataset.data_ranges[2], (slice(int(max_h / 2), max_h, None), slice(0, int(max_w / 2), None)))
        self.assertEqual(self.dataset.data_ranges[3], (slice(int(max_h / 2), max_h, None), slice(int(max_w / 2), max_w, None)))

        overlap = 20
        self.dataset.set_data_ranges(1, 1, overlap=overlap)
        self.assertEqual(len(self.dataset.data_ranges), 4)
        self.assertEqual(self.dataset.data_ranges[0], (slice(0, int(max_h / 2) + overlap, None), slice(0, int(max_w / 2) + overlap, None)))
        self.assertEqual(self.dataset.data_ranges[1], (slice(0, int(max_h / 2) + overlap, None), slice(int(max_w / 2) - overlap, max_w, None)))
        self.assertEqual(self.dataset.data_ranges[2], (slice(int(max_h / 2) - overlap, max_h, None), slice(0, int(max_w / 2) + overlap, None)))
        self.assertEqual(self.dataset.data_ranges[3], (slice(int(max_h / 2) - overlap, max_h, None), slice(int(max_w / 2) - overlap, max_w, None)))


class RadarDataLoaderConfigTests(unittest.TestCase):
    def setUp(self):
        from radar import RadarDatasetFolder
        self.root = Path("/home/eivind/Documents/polarlys_datasets")

    def test_valid_config(self):
        cfg = self.root / "test" / "polarlys_cfg_test.txt"
        self.dataset = RadarDatasetFolder(str(self.root), cfg=str(cfg), split="train", dataset_name="test")
        self.assertEqual(self.dataset.class_names, )

if __name__ == '__main__':
    from radar import RadarDatasetFolder
    unittest.main()