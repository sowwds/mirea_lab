import React from 'react';
import { emotionToPlanetIndex, getEmotionColor, getEmotionData } from '../services/emotionData';

const BottomTab = ({ focusOnPlanet, randomEmotion }) => {
  // Используем randomEmotion из пропсов
  const emotionData = randomEmotion ? getEmotionData(randomEmotion) : { name: 'Неизвестно' };
  const planetName = emotionData.name;
  const planetColor = randomEmotion ? getEmotionColor(randomEmotion, 1) : 'rgba(255, 255, 255, 1)';
  const planetIndex = randomEmotion ? emotionToPlanetIndex[randomEmotion] - 1 : -1;

  // Обработчик клика для вызова focusOnPlanet
  const handlePlanetClick = () => {
    if (planetIndex !== -1) {
      focusOnPlanet(planetIndex);
    }
  };

  return (
    <div className="block sm:hidden w-full h-32 backdrop-blur-3xl bg-surface0/20 rounded-t-4xl border-t border-surface0 absolute bottom-0 z-20 px-6 py-4">
      <h2 className="text-base text-center font-semibold text-white mb-2">Добрый день!</h2>
      <div className="bg-white/2 rounded-md p-2">
        <p className="text-white/80 text-sm leading-relaxed">
          Сегодня мы советуем вам обратить внимание на{' '}
          <span
            className="font-semibold cursor-pointer"
            style={{ color: planetColor }}
            onClick={handlePlanetClick}
          >
            {planetName}
          </span>
        </p>
      </div>
    </div>
  );
};

export default BottomTab;
