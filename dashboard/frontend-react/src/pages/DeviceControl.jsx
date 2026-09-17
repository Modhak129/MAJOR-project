import React from 'react';
import DeviceManager from '../components/DeviceManager';
import AlertFeed from '../components/AlertFeed';

export default function DeviceControl({ socket }) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <DeviceManager />
      <AlertFeed socket={socket} />
    </div>
  );
}
