import unittest

from rafael.network import base as net


class HTTPHandlerTestCase(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.profile = {
            "node_id": 1,
            "protocol": "http",
            "host": "127.0.0.1",
            "subpath": "/group-prs",
            "port": 8000
        }
        self.api = "foo"
        self.args = (1, 2.0, "a")

    def test_construct_url(self):
        ans = "http://127.0.0.1:8000/group-prs/tasks"
        self.assertEqual(ans, net.construct_url(self.profile, "/tasks"))

    def test_construct_request(self):
        ans = {
            "node_id": "1",
            "api": self.api,
            "args": self.args,
        }
        self.assertEqual(ans, net.construct_request(self.profile, self.api, self.args))
