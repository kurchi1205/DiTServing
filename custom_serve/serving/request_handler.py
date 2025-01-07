
import uuid
from datetime import datetime

class RequestPool:
    def __init__(self):
        self.requests = {}

    def add_request(self, request):
        self.requests[request["request_id"]] = request

    def get_all_requests(self):
        return list(self.requests.values())
    
    def remove_request(self, request_id):
       if request_id in self.requests:
            del self.requests[request_id]

    def validate_request(self, request):
        required_fields = ["request_id", "prompt", "timestamp", "status", "timesteps_left"]
        for field in required_fields:
            if field not in request:
                return False
        return True

    def get_valid_requests(self):
        return {req_id: req for req_id, req in self.requests.items() if self.validate_request(req)}


class RequestHandler:
    def __init__(self):
        self.request_pool = RequestPool()

    def create_request(self, prompt, timesteps_left):
        request = {
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "status": "pending",
            "timesteps_left": timesteps_left,
            "prompt": prompt
        }
        return request
    
    def add_request(self, prompt, timesteps_left):
        request = self.create_request(prompt, timesteps_left)
        self.request_pool.add_request(request)

    def get_requests(self):
        return self.request_pool.get_valid_requests()

    def remove_invalid_requests(self):
        self.request_pool.requests = {
            req_id: req for req_id, req in self.request_pool.requests.items() if self.request_pool.validate_request(req)
        }

    def update_timesteps_left(self, request_id):
        if request_id in self.request_pool.requests:
            self.request_pool.requests[request_id]["timesteps_left"] -= 1

    def update_status(self, request_id):
        if request_id in self.request_pool.requests:
            request = self.request_pool.requests[request_id]
            if request["timesteps_left"] == 0:
                request["status"] = "completed"
            else:
                request["status"] = "in_progress"

    