import { useEffect, useRef, useState } from 'react';
import { io, Socket } from 'socket.io-client';
import { useDashboard } from '../context/DashboardContext';
import type { AlertEvent } from '../types';

export function useWebSocket() {
  const socketRef = useRef<Socket | null>(null);
  const { addEvents } = useDashboard();
  const [isConnected, setIsConnected] = useState(false);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);

  useEffect(() => {
    // Connect to the Flask-SocketIO server
    const socket = io('http://localhost:5002', {
      path: '/socket.io',
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 10000,
      reconnectionAttempts: Infinity,
    });

    socketRef.current = socket;

    socket.on('connect', () => {
      console.log('WebSocket connected to CYUKTI backend');
      setIsConnected(true);
      setReconnectAttempts(0);

      // Subscribe to event stream
      socket.emit('subscribe_events', { client: 'dashboard' }, (response: any) => {
        console.log('Subscribed to events:', response);
      });
    });

    socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      setIsConnected(false);
    });

    socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error.message);
      setReconnectAttempts(prev => prev + 1);
    });

    socket.on('new_events', (data: { events: AlertEvent[] }) => {
      if (data && data.events && data.events.length > 0) {
        console.log(`Received ${data.events.length} new event(s) via WebSocket`);
        addEvents(data.events);
      }
    });

    socket.on('response_status_update', (data: any) => {
      console.log('Response status update:', data);
      // Handle response status updates (e.g., BLOCKED, UNBLOCKED)
      // This could trigger a UI notification or update event display
    });

    return () => {
      socket.disconnect();
    };
  }, [addEvents]);

  return { isConnected, reconnectAttempts, socket: socketRef.current };
}
