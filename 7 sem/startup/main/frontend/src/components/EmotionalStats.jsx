import React, { useRef, useEffect, useState, useMemo } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader';
import { planetConfigs } from './planetLoader';
import { emotionsDataByTime, emotionsList, emotionToPlanetIndex, getGradient, getEmotionColor } from '../services/emotionData';

// Глобальный кэш для изображений
const imageCache = {};

// Компонент для анимированного прогресс-бара
const AnimatedProgressBar = ({ value, gradient, emotionKey, timeRange }) => {
  const [animatedValue, setAnimatedValue] = useState(0);

  useEffect(() => {
    // Сбрасываем анимацию при изменении периода
    setAnimatedValue(0);
    const timer = setTimeout(() => {
      setAnimatedValue(value);
    }, 50);

    return () => clearTimeout(timer);
  }, [value, timeRange]);

  return (
    <div className="bg-white/10 rounded-full h-3 relative overflow-hidden">
      <div
        className="h-3 rounded-full transition-all duration-700 ease-out"
        style={{
          width: `${animatedValue}%`,
          background: gradient
        }}
      />
    </div>
  );
};

// Компонент для рендеринга .glb модели в 2D-изображение
const PlanetIcon2D = ({ planet, index, onClick, renderer }) => {
  const loadPath = planet.emotion === 'sadness' && planet.iconPath ? planet.iconPath : planet.modelPath;
  const [imageUrl, setImageUrl] = useState(imageCache[loadPath]);
  const [isLoading, setIsLoading] = useState(!imageCache[loadPath]);

  useEffect(() => {
    if (imageCache[loadPath]) {
      setImageUrl(imageCache[loadPath]);
      setIsLoading(false);
      return;
    }

    const loader = new GLTFLoader();
    const dracoLoader = new DRACOLoader();
    dracoLoader.setDecoderPath('/draco/');
    loader.setDRACOLoader(dracoLoader);

    console.log(`Загрузка модели для ${planet.name}: ${loadPath}`);
    loader.load(
      loadPath,
      (gltf) => {
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 1000);
        camera.position.set(0, 0, 1.5);

        const model = gltf.scene;

        // ДЛЯ ГРУСТИ - ФИЛЬТРУЕМ ТОЛЬКО ОСНОВНУЮ ПЛАНЕТУ (БЕЗ КАМНЕЙ/ОСКОЛКОВ)
        let targetModel = model;

        if (planet.emotion === 'sadness') {
          console.log('🔧 Обработка модели грусти - поиск основной планеты...');

          // Ищем самую большую меш-группу (это должна быть основная планета)
          let mainPlanet = null;
          let maxSize = 0;

          model.traverse((child) => {
            if (child.isMesh || child.isGroup) {
              const box = new THREE.Box3().setFromObject(child);
              const size = box.getSize(new THREE.Vector3());
              const volume = size.x * size.y * size.z;

              if (volume > maxSize) {
                maxSize = volume;
                mainPlanet = child;
              }
            }
          });

          if (mainPlanet) {
            targetModel = mainPlanet;
            console.log(`✅ Найдена основная планета грусти, объем: ${maxSize.toFixed(2)}`);
          } else {
            console.log('⚠️ Основная планета не найдена, используем всю модель');
          }
        }

        // КЛОНИРУЕМ МОДЕЛЬ, ЧТОБЫ НЕ ИЗМЕНЯТЬ ОРИГИНАЛ
        const clonedModel = targetModel.clone();
        clonedModel.position.set(0, 0, 0);
        clonedModel.rotation.set(0, 0, 0);
        clonedModel.scale.set(1, 1, 1);

        // ВЫЧИСЛЯЕМ BOUNDING BOX ТОЛЬКО ДЛЯ ОСНОВНОЙ ЧАСТИ
        const box = new THREE.Box3().setFromObject(clonedModel);
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);

        const center = new THREE.Vector3();
        box.getCenter(center);

        console.log(`📏 ${planet.name}: размеры =`, {
          x: size.x.toFixed(2),
          y: size.y.toFixed(2),
          z: size.z.toFixed(2),
          maxDim: maxDim.toFixed(2),
          center: { x: center.x.toFixed(2), y: center.y.toFixed(2), z: center.z.toFixed(2) }
        });

        // МАСШТАБИРОВАНИЕ
        const TARGET_SIZE = 1.0;
        const baseScale = TARGET_SIZE / maxDim;

        // ИНДИВИДУАЛЬНЫЕ КОРРЕКТИРОВКИ
        const scaleCorrections = {
          sadness: 0.8,
          fear: 0.8,
          disgust: 0.8,
          joy: 0.8,
          trust: 0.8,
          anger: 0.8,
          surprise: 0.8,
          anticipation: 0.8
        };

        const finalScale = baseScale * (scaleCorrections[planet.emotion] || 1.0);

        // ЦЕНТРИРОВАНИЕ
        clonedModel.position.set(-center.x * finalScale, -center.y * finalScale, -center.z * finalScale);
        clonedModel.scale.set(finalScale, finalScale, finalScale);

        scene.add(clonedModel);

        // ОСВЕЩЕНИЕ
        const ambientLight = new THREE.AmbientLight(0xffffff, 1);
        // scene.add(ambientLight);

        const directionalLight1 = new THREE.DirectionalLight(0xFFFFFF, 1);
        directionalLight1.position.set(1, 1, 1);
        scene.add(directionalLight1);

        const directionalLight2 = new THREE.DirectionalLight(0x72bbecff, 1);
        directionalLight2.position.set(-2, -1, -1);
        scene.add(directionalLight2);

        // РЕНДЕРИНГ
        renderer.setSize(64, 64);
        renderer.setClearColor(0x000000, 0);
        renderer.render(scene, camera);

        const imageData = renderer.domElement.toDataURL('image/png');
        imageCache[loadPath] = imageData;
        setImageUrl(imageData);
        setIsLoading(false);
        console.log(`✅ 2D-изображение для ${planet.name} создано`, {
          scale: finalScale.toFixed(3),
          correction: scaleCorrections[planet.emotion] || 1.0
        });

        // Очистка
        scene.remove(clonedModel, ambientLight, directionalLight1, directionalLight2);
      },
      undefined,
      (error) => {
        console.error(`❌ Ошибка загрузки модели для ${planet.name}:`, error);
        createFallbackImage(planet.emotion);
      }
    );

    return () => {
      dracoLoader.dispose();
    };
  }, [planet.modelPath, planet.iconPath, planet.emotion, planet.name, renderer, loadPath]);

  const createFallbackImage = (emotion) => {
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');

    // Градиентный фон вместо простого цвета
    const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    gradient.addColorStop(0, getEmotionColor(emotion, 1));
    gradient.addColorStop(1, getEmotionColor(emotion, 0.3));

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 64, 64);

    // Белая точка в центре
    ctx.fillStyle = 'white';
    ctx.beginPath();
    ctx.arc(32, 32, 8, 0, 2 * Math.PI);
    ctx.fill();

    const imageData = canvas.toDataURL('image/png');
    imageCache[loadPath] = imageData;
    setImageUrl(imageData);
    setIsLoading(false);
  };

  if (isLoading) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-gray-800 rounded-full">
        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <img
      src={imageUrl}
      alt={planet.name}
      className="w-full h-full rounded-full object-contain cursor-pointer hover:scale-110"
      style={{
        backgroundColor: 'transparent',
        transition: 'background-color 0.2s, box-shadow 0.2s, transform 0.3s ease-in-out',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.backgroundColor = getEmotionColor(planet.emotion, 0.5);
        e.currentTarget.style.boxShadow = `0 0 15px ${getEmotionColor(planet.emotion, 0.5)}`;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.backgroundColor = 'transparent';
        e.currentTarget.style.boxShadow = 'none';
      }}
      onClick={() => {
        console.log(`🎯 Клик по иконке ${planet.name}, индекс: ${index}`);
        onClick(index);
      }}
      onMouseOver={() => (document.body.style.cursor = 'pointer')}
      onMouseOut={() => (document.body.style.cursor = 'auto')}
      onError={(e) => {
        console.warn(`🖼️ Ошибка загрузки изображения для ${planet.name}`);
        createFallbackImage(planet.emotion);
      }}
    />
  );
};

const EmotionalStats = ({ isVisible, isMobile, onClose, focusOnPlanet }) => {
  const modalRef = useRef(null);
  const [timeRange, setTimeRange] = useState('week');
  const buttonRefs = useRef({});
  const [prevTimeRange, setPrevTimeRange] = useState('week');

  // Единый рендерер для всех иконок (рендерим последовательно)
  const renderer = useMemo(() => {
    const r = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'low-power' // Экономия энергии
    });
    r.setSize(64, 64);
    r.setPixelRatio(1); // Фиксируем пиксельное соотношение для производительности
    return r;
  }, []);

  // Очистка рендерера
  useEffect(() => {
    return () => {
      setTimeout(() => {
        renderer.dispose();
      }, 1000); // Даем время завершить рендеринг
    };
  }, [renderer]);

  // Закрытие по клику вне модального окна
  useEffect(() => {
    const handleClickOutside = (event) => {
      const statsButton = document.querySelector('[data-stats-button="true"]');
      if (modalRef.current && !modalRef.current.contains(event.target) && statsButton && !statsButton.contains(event.target)) {
        onClose();
      }
    };

    if (isVisible) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isVisible, onClose]);

  // Обновляем предыдущий период при изменении
  useEffect(() => {
    if (timeRange !== prevTimeRange) {
      setPrevTimeRange(timeRange);
    }
  }, [timeRange, prevTimeRange]);

  const overallBalance = emotionsList.reduce((sum, key) => sum + emotionsDataByTime[key][timeRange].value, 0) / emotionsList.length;
  const dominantEmotion = emotionsList.reduce((prev, key) => {
    const current = emotionsDataByTime[key][timeRange];
    return current.value > prev.value ? { name: emotionsDataByTime[key].name, value: current.value } : prev;
  }, { value: 0 });

  const getPlanetData = (emotionKey) => {
    const planetIndex = emotionToPlanetIndex[emotionKey];
    return planetConfigs[planetIndex] || {
      name: emotionKey,
      modelPath: '/models/placeholder.glb',
      emotion: emotionKey
    };
  };

  // Корректировка отступа для десктоп-версии, чтобы не ближе 42px от верха
  useEffect(() => {
    if (isVisible && !isMobile && modalRef.current) {
      const vh = window.innerHeight;
      const mh = modalRef.current.offsetHeight;
      const minTop = 42 + mh / 2; // Чтобы верхний край >= 42px
      const centeredTop = vh / 2;
      const calculatedTop = Math.max(minTop, centeredTop);
      modalRef.current.style.top = `${calculatedTop}px`;
    }
  }, [isVisible, isMobile]);

  // Мобильная версия
  if (isMobile) {
    return (
      <>
        <div className={`fixed inset-0 bg-black/70 backdrop-blur-sm z-1000 transition-all duration-300 ${
          isVisible ? 'opacity-100' : 'opacity-0 pointer-events-none'
        }`} onClick={(e) => {
          const statsButton = document.querySelector('[data-stats-button="true"]');
          if (!statsButton || !statsButton.contains(e.target)) onClose();
        }} />

        <div
          ref={modalRef}
          className={`fixed top-18 left-1/2 z-1000 bg-black/20 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl shadow-blue-500/20 w-80 max-w-[90%] transition-all duration-300 ${
            isVisible ? 'opacity-100 translate-x-[-50%]' : 'opacity-0 translate-x-[-150%]'
          }`}
        >
          <div className="transition-all duration-300">
            <div className="p-4 space-y-3 max-h-[calc(100vh-96px)] overflow-y-auto">
              <h3 className="text-white font-bold text-base uppercase tracking-wide text-center">Эмоциональный профиль</h3>
              <div className="h-px bg-white/20" />

              <div className="space-y-2">
                {/* ... (статистика, если есть) ... */}
              </div>

              <div className="relative flex gap-1 justify-center">
                <div
                  className="absolute h-7 rounded-md bg-gradient-to-r from-white/20 to-white/40 transition-all duration-300 ease-in-out"
                  style={{
                    left: buttonRefs.current[timeRange]?.offsetLeft || 0,
                    width: buttonRefs.current[timeRange]?.offsetWidth || 0,
                    top: 0,
                    boxShadow: '0 0 8px rgba(255, 255, 255, 0.3)',
                  }}
                />
                {['day', 'week', 'month', 'all'].map((range) => (
                  <button
                    key={range}
                    ref={(el) => (buttonRefs.current[range] = el)}
                    onClick={() => setTimeRange(range)}
                    className={`relative px-2 py-1.5 rounded-md text-xs transition-colors duration-300 ${
                      timeRange === range ? 'text-white font-medium' : 'text-white/60 hover:text-white'
                    }`}
                  >
                    {range === 'day' ? 'День' : range === 'week' ? 'Неделя' : range === 'month' ? 'Месяц' : 'Все время'}
                  </button>
                ))}
              </div>

              <div className="space-y-2">
                {emotionsList.map((key, index) => {
                  const emotion = emotionsDataByTime[key];
                  const data = emotion[timeRange];
                  const planet = getPlanetData(key);

                  return (
                    <div key={`${key}-${timeRange}`} className="flex items-center space-x-2 h-10">
                      <div className="w-10 h-10 relative min-w-[2.5rem]">
                        <PlanetIcon2D
                          planet={planet}
                          index={emotionToPlanetIndex[key] - 1}
                          renderer={renderer}
                          onClick={() => {
                            onClose();
                            focusOnPlanet(emotionToPlanetIndex[key] - 1);
                          }}
                        />
                      </div>
                      <div className="flex-1 flex flex-col justify-between min-w-0">
                        <div className="flex justify-between items-center">
                          <span className="text-white/80 text-sm mb-1 truncate">{emotion.name}</span>
                          <div className="flex items-center space-x-2 flex-shrink-0">
                            <span className="text-white text-sm font-medium">{data.value}%</span>
                            <span className={`text-sm ${
                              data.trend === 'up' ? 'text-green-400' : data.trend === 'down' ? 'text-red-400' : 'text-blue-400'
                            }`}>
                              {data.trend === 'up' ? '↑' : data.trend === 'down' ? '↓' : '→'}
                            </span>
                          </div>
                        </div>
                        <AnimatedProgressBar
                          value={data.value}
                          gradient={getGradient(key)}
                          emotionKey={key}
                          timeRange={timeRange}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </>
    );
  }

  // Десктоп версия
  return (
    <>
      <div className={`fixed inset-0 z-1000 transition-all duration-300 ${
        isVisible ? 'opacity-100' : 'opacity-0 pointer-events-none'
      }`} onClick={(e) => {
        const statsButton = document.querySelector('[data-stats-button="true"]');
        if (!statsButton || !statsButton.contains(e.target)) onClose();
      }} />

      <div
        ref={modalRef}
        className={`fixed left-5 z-1010 bg-black/20 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl shadow-blue-500/20 w-80 transition-all duration-300 ${
          isVisible ? 'translate-y-0 opacity-100' : '-translate-x-full opacity-0'
        }`}
        style={{ transform: isVisible ? 'translateY(-50%)' : 'translateX(-100%) translateY(-50%)' }}
      >
        <div className="transition-all duration-300">
          <div className="p-4 space-y-3 max-h-[calc(100vh-96px)] overflow-y-auto">
            <h3 className="text-white font-bold text-base uppercase tracking-wide text-center">Эмоциональный профиль</h3>
            <div className="h-px bg-white/20" />

            <div className="space-y-2">
              {/* ... (статистика, если есть) ... */}
            </div>

            <div className="relative flex gap-1 justify-center">
              <div
                className="absolute h-7 rounded-md bg-gradient-to-r from-white/20 to-white/40 transition-all duration-300 ease-in-out"
                style={{
                  left: buttonRefs.current[timeRange]?.offsetLeft || 0,
                  width: buttonRefs.current[timeRange]?.offsetWidth || 0,
                  top: 0,
                  boxShadow: '0 0 8px rgba(255, 255, 255, 0.3)',
                }}
              />
              {['day', 'week', 'month', 'all'].map((range) => (
                <button
                  key={range}
                  ref={(el) => (buttonRefs.current[range] = el)}
                  onClick={() => setTimeRange(range)}
                  className={`relative px-2 py-1.5 rounded-md text-xs transition-colors duration-300 ${
                    timeRange === range ? 'text-white font-medium' : 'text-white/60 hover:text-white'
                  }`}
                >
                  {range === 'day' ? 'День' : range === 'week' ? 'Неделя' : range === 'month' ? 'Месяц' : 'Все время'}
                </button>
              ))}
            </div>

            <div className="space-y-2">
              {emotionsList.map((key, index) => {
                const emotion = emotionsDataByTime[key];
                const data = emotion[timeRange];
                const planet = getPlanetData(key);

                return (
                  <div key={`${key}-${timeRange}`} className="flex items-center space-x-2 h-10">
                    <div className="w-10 h-10 relative min-w-[2.5rem]">
                      <PlanetIcon2D
                        planet={planet}
                        index={emotionToPlanetIndex[key] - 1}
                        renderer={renderer}
                        onClick={() => {
                          onClose();
                          focusOnPlanet(emotionToPlanetIndex[key] - 1);
                        }}
                      />
                    </div>
                    <div className="flex-1 flex flex-col justify-between min-w-0">
                      <div className="flex justify-between items-center">
                        <span className="text-white/80 text-sm mb-1 truncate">{emotion.name}</span>
                        <div className="flex items-center space-x-2 flex-shrink-0">
                          <span className="text-white text-sm font-medium">{data.value}%</span>
                          <span className={`text-sm ${
                            data.trend === 'up' ? 'text-green-400' : data.trend === 'down' ? 'text-red-400' : 'text-blue-400'
                          }`}>
                            {data.trend === 'up' ? '↑' : data.trend === 'down' ? '↓' : '→'}
                          </span>
                        </div>
                      </div>
                      <AnimatedProgressBar
                        value={data.value}
                        gradient={getGradient(key)}
                        emotionKey={key}
                        timeRange={timeRange}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default EmotionalStats;
