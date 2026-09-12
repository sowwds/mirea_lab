import React from 'react';

const SentMessage = ({ text }) => {
  console.log('SentMessage: Rendering with text', text);

  return (
    <div className="flex justify-end mb-4">
      <div className="backdrop-blur-xl shadow-gray-900/30 shadow-md bg-gray-400/30 text-white text-text py-3 px-4 rounded-2xl max-w-xs whitespace-pre-wrap">
        {text}
      </div>
    </div>
  );
};

export default SentMessage;
