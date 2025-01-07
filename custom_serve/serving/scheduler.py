class Scheduler:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def prioritize_requests(self, request_pool):
        expiring_requests = []
        new_requests = []

        for request in request_pool.values():
            if "cache_interval" in request:
                # Check if the latent is expiring
                if request["cache_interval"] == 0:
                    expiring_requests.append(request)
            else:    
                new_requests.append(request)

        if len(expiring_requests) == 0 and len(new_requests) == 0:
            expiring_requests = request_pool.values()
            expiring_requests.sort(key=lambda req: req["cache_interval"])

        prioritized_requests = expiring_requests + new_requests
        return prioritized_requests

    def create_batch(self, prioritized_requests):
        return prioritized_requests[:self.batch_size]
    
    def schedule(self, request_pool):
        prioritized_requests = self.prioritize_requests(request_pool)
        return self.create_batch(prioritized_requests)
