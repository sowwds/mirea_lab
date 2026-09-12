import React from 'react';

const ReceivedMessage = ({ text }) => {
  // console.log('ReceivedMessage: Rendering with text', text);
  // Разбиваем текст по двойным переносам строк (\n\n)
  const messages = text.split('\n\n').map((message, index, array) => {
    // Определяем классы радиусов углов в зависимости от позиции
    const isSingle = array.length === 1;
    const isFirst = index === 0;
    const isLast = index === array.length - 1;

    const borderRadiusClass = isSingle
      ? 'rounded-2xl'
      : [
          'rounded-l-md rounded-r-2xl', // По умолчанию: левая сторона lg, правая 2xl
          isFirst ? 'rounded-tl-2xl' : '', // Первый: левый верхний угол 2xl
          isLast ? 'rounded-bl-2xl' : '', // Последний: левый нижний угол 2xl
        ]
          .filter(Boolean)
          .join(' ');

    return (
      <div
        key={index}
        className={`backdrop-blur-xl shadow-gray-400/30 shadow-md bg-gray-900/30 text-white py-3 px-4 ${borderRadiusClass} w-fit max-w-xs whitespace-pre-wrap`}
      >
        {message}
      </div>
    );
  });

  return (
    <div className="flex justify-start mb-4 flex-col gap-2">
      {messages}
    </div>
  );
};

export default ReceivedMessage;
