from fastapi import HTTPException


class NodeIdentity:

    def __init__(self, node_id) -> None:
        self.__node_id = str(node_id)

    @property
    def node_id(self):
        return self.__node_id

    def equals_node_id(self, node_id):
        node_id = str(node_id)
        return node_id == self.node_id

    def authenticate(self, node_id):
        node_id = str(node_id)
        if not self.equals_node_id(node_id):
            raise HTTPException(status_code=403, detail="Node ID is inconsistent")


class Authenticator:

    def __init__(self, node_id, group_id) -> None:
        self.__node_id = str(node_id)
        self.__group_id = str(group_id)

    @property
    def node_id(self):
        return self.__node_id

    @property
    def group_id(self):
        return self.__group_id

    def equals_node_id(self, node_id):
        node_id = str(node_id)
        return node_id == self.node_id

    def equals_group_id(self, group_id):
        group_id = str(group_id)
        return group_id == self.group_id

    def authenticate(self, node_id, group_id):
        if not self.equals_node_id(node_id):
            raise HTTPException(status_code=403, detail="Node ID is inconsistent")

        if not self.equals_group_id(group_id):
            raise HTTPException(status_code=403, detail="Group ID is inconsistent")
