import httpx

from openalpha_brain.services.http_pool import get_client


class TestHttpPool:
    def test_get_client_returns_client(self):
        client = get_client()
        assert isinstance(client, httpx.AsyncClient)
        assert not client.is_closed

    def test_get_client_returns_same_instance(self):
        client1 = get_client()
        client2 = get_client()
        assert client1 is client2
