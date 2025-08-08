class CacheManager:
    def __init__(self, cache_interval):
        self.cache_interval = cache_interval

    def update_latent(self, request_pool, request_id, new_latent):
        if request_id in request_pool:
            request_pool[request_id]["latent"] = new_latent
            request_pool[request_id]["cache_interval"] = self.cache_interval

    def update_intervals(self, request_pool):
        for request_id in request_pool:
            request_pool[request_id]["cache_interval"] -= 1
