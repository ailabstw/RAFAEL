import unittest

import rafael.data_repository.repository as repository


class RepositoryTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.repo = repository.InMemoryRepository()
        self.repo.add("A", 1)
        self.repo.add("B", [1, 2, 3])
        self.repo.add("C", "cat")

    def tearDown(self) -> None:
        self.repo = None

    def test_get(self):
        self.assertEqual(self.repo.get("A"), 1)
        self.assertEqual(self.repo.get("B"), [1, 2, 3])
        self.assertEqual(self.repo.get("C"), "cat")
        self.assertEqual(self.repo.get("D"), None)
    
    def test_list(self):
        self.assertEqual(self.repo.list(), ["A", "B", "C"])
        
    def test_pop(self):
        self.assertEqual(self.repo.pop("A"), 1)
        try:
            self.repo.pop("A")
            raise AssertionError
        except:
            assert KeyError
            
    def test_indexing(self):
        self.assertEqual(self.repo["A"], 1)
        self.assertEqual(self.repo["B"], [1, 2, 3])
        self.assertEqual(self.repo["C"], "cat")
        try:
            self.repo["D"]
            raise AssertionError
        except:
            assert KeyError
