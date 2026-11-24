import asyncio
import socket
import time
from typing import Set, Dict, Callable, List
from common.logging import info, trace, error
from assistant.AssistantEvents import observers


DRAIN_TIMEOUT = 5
# What a TCP outpost is sent for a command only the in-process listeners know.
TCP_COMMANDS = {"resume": "wakeword"}


class OutpostServer:
    """TCP server that pushes stop/listen/wakeword commands to every connected microphone outpost.

    In-process listeners (local mic outposts) receive the same commands via add_command_listener,
    so they work even when no TCP client is connected.
    """

    def __init__(self, port: int = 43523):
        self.port = port
        self.server = None
        self.clients: Set[asyncio.StreamWriter] = set()
        self.running = False
        self.ping_timestamps: Dict[asyncio.StreamWriter, float] = {}
        self.loop: asyncio.AbstractEventLoop | None = None
        # In-process subscribers (e.g. local mic outposts) that receive the same
        # commands as TCP-connected outposts, even when no TCP client is connected.
        self.command_listeners: List[Callable[[str], None]] = []

    def add_command_listener(self, listener: Callable[[str], None]):
        self.command_listeners.append(listener)

    async def start(self):
        """Start the TCP server for outpost notifications"""
        self.loop = asyncio.get_running_loop()
        try:
            self.server = await asyncio.start_server(
                self.handle_client,
                '0.0.0.0',
                self.port,
                reuse_address=True
            )
            self.running = True

            addrs = ', '.join(str(sock.getsockname()) for sock in self.server.sockets)
            trace(f'Outpost server listening on {addrs}')

            async with self.server:
                await self.server.serve_forever()

        except Exception as e:
            info(f"Failed to start Outpost server: {e}")
            raise

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a new client connection"""
        addr = writer.get_extra_info('peername')
        info(f"Outpost connected from {addr}")

        # Configure socket for low latency
        sock = writer.get_extra_info('socket')
        if sock:
            # Disable Nagle's algorithm for lower latency
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # Set socket to non-blocking mode
            sock.setblocking(False)

        # Add to active clients
        self.clients.add(writer)

        # Ping throttling: only send ping every 100 timeouts (5 seconds with 50ms timeout)
        timeout_counter = 0

        try:
            # Keep connection alive and handle any incoming data
            while True:
                try:
                    # Read data with short timeout (50ms) for responsive command delivery
                    data = await asyncio.wait_for(reader.read(1024), timeout=0.05)  # TODO THIS IS BULLSHIT, clean this server up
                    if not data:
                        break

                    # Reset timeout counter since we received data
                    timeout_counter = 0

                    # Handle incoming messages from client
                    message = data.decode('utf-8').strip()
                    if message == "<pong>":
                        # Calculate round trip time
                        if writer in self.ping_timestamps:
                            rtt = (time.time() - self.ping_timestamps[writer]) * 1000  # Convert to milliseconds
                            observers().on_outpost_latency(addr, rtt)
                            del self.ping_timestamps[writer]
                except asyncio.TimeoutError:
                    # Only send ping every 100 timeouts (5 seconds)
                    timeout_counter += 1
                    if timeout_counter >= 100:
                        try:
                            self.ping_timestamps[writer] = time.time()
                            writer.write(b"<ping>\n")
                            await asyncio.wait_for(writer.drain(), timeout=DRAIN_TIMEOUT)
                            timeout_counter = 0
                        except (asyncio.TimeoutError, Exception) as e:
                            trace(f"Ping to Outpost {addr} failed: {e}")
                            break
                except Exception:
                    break

        except Exception as e:
            error(f"Error handling Outpost {addr}: {e}")
        finally:
            # Remove from active clients
            self.clients.discard(writer)
            # Clean up ping timestamp if exists
            if writer in self.ping_timestamps:
                del self.ping_timestamps[writer]
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            info(f"Client disconnected from {addr}")

    async def broadcast_command(self, command):
        """Send command to all connected clients"""
        # In-process listeners (local mic outposts) get every command regardless of
        # whether any TCP outpost is connected.
        for listener in self.command_listeners:
            try:
                listener(command)
            except Exception as e:
                error(f"Local command listener failed for '{command}': {e}")

        if not self.clients:
            trace("No TCP-connected Outposts to notify")
            return

        # A TCP outpost knows 'stop', 'listen' and 'wakeword' alone.
        command = TCP_COMMANDS.get(command, command)

        # Create a copy of clients set to avoid modification during iteration
        clients_copy = self.clients.copy()
        disconnected_clients = set()

        for client in clients_copy:
            try:
                client.write(f"<{command}>\n".encode())
                await asyncio.wait_for(client.drain(), timeout=DRAIN_TIMEOUT)
                trace(f"Sent <{command}> command to Outpost {client.get_extra_info('peername')}")
            except asyncio.TimeoutError:
                error(f"Timeout sending to Outpost {client.get_extra_info('peername')}, dropping client")
                disconnected_clients.add(client)
            except Exception as e:
                error(f"Failed to send to Outpost {client.get_extra_info('peername')}: {e}")
                disconnected_clients.add(client)

        # Remove disconnected clients
        for client in disconnected_clients:
            self.clients.discard(client)
            try:
                client.close()
                await client.wait_closed()
            except Exception:
                pass

    async def stop(self):
        """Stop the server and close all client connections"""
        self.running = False

        # Close all client connections
        for client in self.clients.copy():
            try:
                client.close()
                await client.wait_closed()
            except Exception:
                pass

        self.clients.clear()

        # Stop the server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            info("Outpost server stopped")


# Global instance
outpost_server = OutpostServer()
