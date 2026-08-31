import { useEffect, useRef } from 'react';
import { io, Socket } from 'socket.io-client';
import { useDashboard } from '../context/DashboardContext';
import type { AlertEvent } from '../types';

export function useWebSocket() {
  const socketRef = useRef<Socket | null>(null);
  const { addEvents } = useDashboard();

  useEffect(() => {
    // Connect to the Flask-SocketIO server
    const socket = io('/', {
      path: '/socket.io',
      transports: ['websocket', 'polling'],
    });

    socketRef.current = socket;

    socket.on('connect', () => {
      console.log('WebSocket connected');
      socket.emit('subscribe_events', { client: 'dashboard' });
    });

    socket.on('disconnect', () => {
      console.log('WebSocket disconnected');
    });

    socket.on('new_events', (data: { events: AlertEvent[] }) => {
      if (data && data.events && data.events.length > 0) {
        addEvents(data.events);
      }
    });

    return () => {
      socket.disconnect();
    };
  }, [addEvents]);

  return socketRef.current;
}
