CacheScout results produced before 2026-09-29, when the baseline's prefix fingerprint
assigned one agent id to every request of all four programs (anchor = the 48-token
shared chat-template header; the support cliff treated the 4-way program fork as the
end of the fixed context). Its transition chain was a single self-loop and it issued
no prefetch, so these runs are vanilla with a different eviction-band map.
Replaced by the call-site-identity CacheScout (server/cacheScout_server.py docstring).
Kept only for the record; do not cite.
