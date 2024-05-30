import unittest

from rafael.authentication import Authenticator, NodeIdentity


class AuthenticatorTestCase(unittest.TestCase):

    def setUp(self):
        self.auth = Authenticator(1, 1)

    def test_authenticate(self):
        self.assertEqual('1', self.auth.node_id)
        self.assertEqual('1', self.auth.group_id)

        self.assertTrue(self.auth.equals_node_id(1))
        self.assertFalse(self.auth.equals_node_id(2))
        self.assertTrue(self.auth.equals_group_id(1))
        self.assertFalse(self.auth.equals_group_id(2))


class NodeIdentityTestCase(unittest.TestCase):

    def setUp(self):
        self.auth = NodeIdentity(1)

    def test_authenticate(self):
        self.assertEqual('1', self.auth.node_id)

        self.assertTrue(self.auth.equals_node_id(1))
        self.assertFalse(self.auth.equals_node_id(2))
