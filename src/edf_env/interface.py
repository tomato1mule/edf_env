class EdfInterface():
    def __init__(self):
        raise NotImplementedError

    def observe_scene(self, obs_type: str ='pointcloud', update: bool = True):
        raise NotImplementedError

    def observe_ee(self, obs_type: str ='pointcloud', update: bool = True):
        raise NotImplementedError

    def pick(self, poses):
        raise NotImplementedError

    def place(self, poses):
        raise NotImplementedError

