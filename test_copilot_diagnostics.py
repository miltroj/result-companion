"""Diagnostics: explores Copilot SDK startup, status, auth, and ping."""

import asyncio

from copilot import CopilotClient


async def main():
    client = CopilotClient()
    print(f"[state]  {client.get_state()}")

    print("[start]  starting client...")
    await client.start()
    print(f"[state]  {client.get_state()}")

    status = await client.get_status()
    print(f"[status] version={status.version} protocol={status.protocolVersion}")

    auth = await client.get_auth_status()
    print(f"[auth]   authenticated={auth.isAuthenticated} login={auth.login}")

    ping = await client.ping("health check")
    print(f"[ping]   message={ping.message} timestamp={ping.timestamp}")

    await client.stop()
    print(f"[state]  {client.get_state()}")


asyncio.run(main())
