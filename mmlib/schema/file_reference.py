class FileReference:

    def __init__(self, path: str = None, reference_id: str = None):
        self.path = path
        self.reference_id = reference_id
        self.size = None
