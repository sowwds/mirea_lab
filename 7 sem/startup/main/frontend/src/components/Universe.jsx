import React, {useEffect, useMemo, useRef, useState} from 'react';
import * as THREE from 'three';
import { TextureLoader } from 'three';
import gsap from 'gsap';
import debounce from 'lodash.debounce';
import Chat from './Chat';
import Header from './Header';
import EmotionalStats from './EmotionalStats';
import PlanetInterface from './PlanetInterface';
import BottomTab from './BottomTab';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';
import { loadPlanetModels, createPlanetsInScene, getPlanetPositions } from './planetLoader';
import { loadAllNebulas, updateNebulas } from './nebulaLoader';
import Stats from 'three/examples/jsm/libs/stats.module.js'; // Добавляем Stats.js
import { emotionsList } from '../services/emotionData';

const getIsMobile = () => {
  if (typeof window === 'undefined') return false;
  return window.innerWidth <= 768;
};

export default function Universe() {
  const mountRef = useRef(null);
  const [cameraState, setCameraState] = useState({ mode: 'initial' });
  const planetsRef = useRef([]);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const observerRef = useRef(null);
  const cameraAnimationRef = useRef(null);
  const cameraPositionRef = useRef({ x: 0, y: 5, z: 7 });
  const cameraRotationRef = useRef({ x: 0, y: 0, z: THREE.MathUtils.degToRad(-30) });
  const lookAtTargetRef = useRef({ x: 0, y: -0.5, z: 0 });
  const textureLoaderRef = useRef(new TextureLoader());
  const isMobileRef = useRef(false);
  const navigate = useNavigate();
  const { token } = useAuth();
  const [showStats, setShowStats] = useState(false);
  const [isMobile, setIsMobile] = useState(getIsMobile());
  const [isLoading, setIsLoading] = useState(true);
  const nebulasRef = useRef([]);
  const [randomEmotion, setRandomEmotion] = useState(null);

  // Новые refs для управления состоянием инициализации
  const isInitializedRef = useRef(false);
  const sceneCleanupRef = useRef(null);
  const clickHandlerRef = useRef(null);

  const loaderStyle = useMemo(() => {
    if (typeof window === 'undefined') return {};

    return {
      position: 'fixed',
      top: 0,
      left: 0,
      width: '100%',
      height: '100%',
      backgroundColor: '#0A0E21',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 9999,
      transition: 'opacity 0.5s ease',
      opacity: isLoading ? 1 : 0,
      pointerEvents: isLoading ? 'auto' : 'none',
      transform: 'none',
      margin: 0,
      padding: 0,
      border: 'none',
      outline: 'none',
      boxSizing: 'border-box'
    };
  }, [isLoading]);

  useEffect(() => {
    const checkMobile = () => {
      const mobile = window.innerWidth <= 768;
      setIsMobile(mobile);
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  // Redirect to /auth if not authenticated
  useEffect(() => {
    if (!token) {
      navigate('/auth', { replace: true });
    }
  }, [token, navigate]);

  useEffect(() => {
      setRandomEmotion(emotionsList[Math.floor(Math.random() * emotionsList.length)]);
    }, []);


  // Update camera position and rotation (УБИРАЕМ renderer.render отсюда!)
  const updateCamera = () => {
    if (!cameraRef.current || !sceneRef.current) return;

    const camera = cameraRef.current;
    camera.position.set(
        cameraPositionRef.current.x,
        cameraPositionRef.current.y,
        cameraPositionRef.current.z
    );

    camera.rotation.set(0, 0, 0);
    camera.lookAt(
        lookAtTargetRef.current.x,
        lookAtTargetRef.current.y,
        lookAtTargetRef.current.z
    );

    camera.rotateX(cameraRotationRef.current.x);
    camera.rotateY(cameraRotationRef.current.y);
    camera.rotateZ(cameraRotationRef.current.z);

    // БЕЗ renderer.render! Рендер только в animate()
  };

  // Focus on a specific planet
  const focusOnPlanet = (index) => {
    console.log(`🎯 Программный фокус на планету с индексом: ${index}`);

    // Ищем планету по индексу
    const planet = planetsRef.current.find(p => p.userData && p.userData.index === index);

    if (planet && planet.userData) {
      console.log(`✅ Найдена планета: ${planet.userData.emotion || planet.userData.name}`);

      const dir = planet.userData.isBigPlanet ? null : new THREE.Vector3();
      if (dir && cameraRef.current) {
        cameraRef.current.getWorldDirection(dir);
        dir.normalize();
      }

      setCameraState({
        mode: 'focused',
        target: planet.userData.target,
        index: planet.userData.index,
        originalPosition: planet.userData.originalPosition,
        direction: dir,
        emotion: planet.userData.emotion
      });
    } else {
      console.warn(`❌ Планета с индексом ${index} не найдена`);
      // Показываем доступные планеты для отладки
      console.log('📋 Доступные планеты:', planetsRef.current.map(p => ({
        index: p.userData?.index,
        emotion: p.userData?.emotion,
        name: p.userData?.name
      })));
    }
  };

  // Create large nebula with fade-out edges
  const createBigNebula = (texture, opacity) => {
    const originalRadius = 2.9;
    const maxRadius = 3.9;
    const segments = 128;
    const geometry = new THREE.RingGeometry(0.1, maxRadius, segments);

    const vertexShader = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

    const fragmentShader = `
uniform sampler2D map;
uniform float opacity;
uniform float originalRadius;
uniform float maxRadius;
varying vec2 vUv;
void main() {
  vec4 texColor = texture2D(map, vUv);
  float dist = 0.1 + vUv.x * (maxRadius - 0.1);
  float fade = 1.0;
  if (dist > originalRadius) {
    fade = pow((maxRadius - dist) / (maxRadius - originalRadius), 1.5);
  }
  gl_FragColor = vec4(texColor.rgb, texColor.a * opacity * fade);
}
`;

    const material = new THREE.ShaderMaterial({
      uniforms: {
        map: { value: texture },
        opacity: { value: opacity },
        originalRadius: { value: originalRadius },
        maxRadius: { value: maxRadius }
      },
      vertexShader,
      fragmentShader,
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false
    });

    const nebula = new THREE.Mesh(geometry, material);
    nebula.rotation.x = Math.PI / 2;
    nebula.position.y = -0.5;
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.repeat.set(1, 1);
    nebula.userData = {
      rotationSpeed: 0.0001
    };
    const customDepthMaterial = new THREE.MeshDepthMaterial({
      depthPacking: THREE.RGBADepthPacking,
      alphaMap: texture,
      opacity: opacity
    });
    nebula.customDepthMaterial = customDepthMaterial;
    nebula.castShadow = true;
    nebula.receiveShadow = true;
    return nebula;
  };

  // Create background star field
  const createStarField = (count = 3000) => {
    const vertices = [];
    const colors = [];
    for (let i = 0; i < count; i++) {
      const x = (Math.random() - 0.5) * 2000;
      const y = (Math.random() - 0.5) * 2000;
      const z = (Math.random() - 0.5) * 2000;
      vertices.push(x, y, z);
      colors.push(1, 1, 1);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    const material = new THREE.PointsMaterial({
      size: 0.6,
      sizeAttenuation: true,
      vertexColors: true,
      transparent: true,
      opacity: 0.8
    });
    return new THREE.Points(geometry, material);
  };

  // Handle window resize with debouncing
  const handleResize = debounce(() => {
    if (!mountRef.current || !cameraRef.current || !rendererRef.current) return;

    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;
    if (width <= 0 || height <= 0) return;

    isMobileRef.current = window.innerWidth <= 768;
    rendererRef.current.setSize(width, height);
    cameraRef.current.fov = isMobileRef.current ? 55 : 40;
    cameraRef.current.aspect = width / height;
    cameraRef.current.updateProjectionMatrix();

    updateCamera();
  }, 100);

  // Функция очистки сцены
  const cleanupScene = () => {
    console.log('🧹 Начало очистки Three.js сцены');

    // Останавливаем все анимации GSAP
    if (cameraAnimationRef.current) {
      cameraAnimationRef.current.kill();
      cameraAnimationRef.current = null;
    }

    // Удаляем обработчики событий
    if (clickHandlerRef.current && rendererRef.current && rendererRef.current.domElement) {
      rendererRef.current.domElement.removeEventListener('click', clickHandlerRef.current);
      clickHandlerRef.current = null;
    }

    // Удаляем ResizeObserver
    if (mountRef.current && observerRef.current) {
      observerRef.current.unobserve(mountRef.current);
      observerRef.current = null;
    }

    // Удаляем обработчик resize
    window.removeEventListener('resize', handleResize);

    // Очищаем Three.js объекты
    if (rendererRef.current) {
      if (mountRef.current && mountRef.current.contains(rendererRef.current.domElement)) {
        mountRef.current.removeChild(rendererRef.current.domElement);
      }
      rendererRef.current.dispose();
      rendererRef.current = null;
    }

    // Очищаем ссылки
    planetsRef.current = [];
    nebulasRef.current = [];
    sceneRef.current = null;
    cameraRef.current = null;

    isInitializedRef.current = false;

    console.log('✅ Three.js сцена полностью очищена');
  };

  // Main scene setup
  useEffect(() => {
    if (!mountRef.current || isInitializedRef.current) {
      console.log('ℹ️ Сцена уже инициализирована или mountRef не готов');
      return;
    }

    console.log('🚀 Инициализация Three.js сцены');
    isInitializedRef.current = true;

    // Сохраняем функцию очистки
    sceneCleanupRef.current = cleanupScene;

    // Очищаем контейнер
    while (mountRef.current.firstChild) {
      mountRef.current.removeChild(mountRef.current.firstChild);
    }

    const scene = new THREE.Scene();
    sceneRef.current = scene;
    isMobileRef.current = window.innerWidth <= 768;

    const initialFov = isMobileRef.current ? 55 : 40;
    const camera = new THREE.PerspectiveCamera(
        initialFov,
        mountRef.current.clientWidth / mountRef.current.clientHeight,
        0.1,
        1000
    );

    cameraRef.current = camera;
    updateCamera();

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true
    });

    rendererRef.current = renderer;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    mountRef.current.appendChild(renderer.domElement);

    // Добавляем Stats.js
    // const stats = Stats();
    // document.body.appendChild(stats.dom);
    // stats.dom.style.position = 'absolute';
    // stats.dom.style.top = '0px';
    // stats.dom.style.zIndex = '10000';

    const starField = createStarField(isMobileRef.current ? 3000 : 4000);
    scene.add(starField);

    loadAllNebulas(textureLoaderRef.current)
        .then((nebulas) => {
          nebulasRef.current = nebulas;
          nebulas.forEach(nebula => {
            nebula.renderOrder = 1;
            scene.add(nebula);
          });
        })
        .catch(error => {
          console.error('Ошибка загрузки туманностей:', error);
        });

    const planetLight = new THREE.PointLight(0xffffff, 10, 20);
    const isLowPerformance = window.innerWidth * window.innerHeight > 1280 * 720 || window.devicePixelRatio > 1.5;
    renderer.shadowMap.enabled = !isLowPerformance;
    planetLight.castShadow = !isLowPerformance;
    planetLight.position.set(0, -0.5, 0);
    planetLight.shadow.mapSize.width = 512;
    planetLight.shadow.mapSize.height = 512;
    planetLight.shadow.camera.near = 0.1;
    planetLight.shadow.camera.far = 50;
    planetLight.shadow.bias = -0.0001;
    planetLight.shadow.normalBias = 0.05;
    scene.add(planetLight);

    // Улучшенный обработчик клика
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const onClick = (event) => {
      if (!cameraRef.current || !sceneRef.current || !planetsRef.current.length) {
        console.warn('❌ Сцена не готова для обработки клика');
        return;
      }

      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, cameraRef.current);

      // Фильтруем только кликабельные объекты
      const clickableObjects = planetsRef.current.filter(planet =>
          planet.userData && planet.userData.target
      );

      console.log(`🎯 Объектов для клика: ${clickableObjects.length}`);

      const intersects = raycaster.intersectObjects(clickableObjects, true);

      if (intersects.length > 0) {
        let obj = intersects[0].object;

        // Ищем родительский объект с userData
        while (obj && !obj.userData?.target && obj.parent) {
          obj = obj.parent;
        }

        if (obj && obj.userData && obj.userData.target) {
          console.log(`✅ Клик по планете: ${obj.userData.emotion || obj.userData.name}, индекс: ${obj.userData.index}`);

          const dir = obj.userData.isBigPlanet ? null : new THREE.Vector3();
          if (dir) {
            cameraRef.current.getWorldDirection(dir);
            dir.normalize();
          }

          setCameraState({
            mode: 'focused',
            target: obj.userData.target,
            index: obj.userData.index,
            originalPosition: obj.userData.originalPosition,
            direction: dir,
            emotion: obj.userData.emotion
          });
        } else {
          console.warn('❌ Объект без userData.target');
        }
      } else {
        console.log('🔍 Пересечений не найдено');
      }
    };

    // Сохраняем ссылку на обработчик
    clickHandlerRef.current = onClick;
    renderer.domElement.addEventListener('click', onClick);

    let animationId;
    const clock = new THREE.Clock();

    const animate = () => {
      animationId = requestAnimationFrame(animate);
      const delta = clock.getDelta(); // Добавляем delta time
      const time = clock.getElapsedTime();

      if (nebulasRef.current && nebulasRef.current.length > 0) {
        updateNebulas(nebulasRef.current, time);
      }

      // Обновляем планеты с delta
      scene.children.forEach(child => {
        if (child.userData && child.userData.update) {
          child.userData.update(delta); // Изменяем на delta вместо time
        }
      });

      // Всегда рендерим здесь
      if (cameraRef.current && rendererRef.current && sceneRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      }

//       stats.update();
    };

    animate();

    // Загрузка планет
    loadPlanetModels(isMobileRef.current).then((planetModels) => {
      console.log('✅ Модели планет загружены:', Object.keys(planetModels));

      const planets = createPlanetsInScene(scene, planetModels, isMobileRef.current);
      planetsRef.current = planets;

      console.log('📝 Создано планет в сцене:', planets.length);
      planets.forEach((planet, index) => {
        if (planet.userData) {
          console.log(`🪐 Планета ${index}:`, {
            emotion: planet.userData.emotion,
            index: planet.userData.index,
            isBigPlanet: planet.userData.isBigPlanet
          });
        }
      });

      const fillLight = new THREE.SpotLight(0xffffff, 3.5, 25, Math.PI / 4, 0.5, 1);
      fillLight.position.set(0, 2, 3);
      fillLight.target = planets[0] || new THREE.Object3D();
      fillLight.castShadow = false;
      scene.add(fillLight);

      setIsLoading(false);
      handleResize();
    }).catch((error) => {
      console.error('❌ Ошибка загрузки моделей планет:', error);
      setIsLoading(false);
      handleResize();
    });

    window.addEventListener('resize', handleResize);
    const observer = new ResizeObserver(handleResize);
    observerRef.current = observer;
    observer.observe(mountRef.current);

    return () => {
      console.log('🔴 Запуск очистки Three.js сцены');
      cancelAnimationFrame(animationId);
//      if (stats.dom.parentNode) {
//        stats.dom.parentNode.removeChild(stats.dom);
//      }
      cleanupScene();
    };
  }, []); // Пустой массив зависимостей - инициализация только один раз

  // Camera focus on planet
  useEffect(() => {
    if (cameraState.mode === 'focused' && cameraState.target && cameraRef.current && sceneRef.current) {
      if (cameraAnimationRef.current) {
        cameraAnimationRef.current.kill();
      }

      const target = cameraState.target;
      const isBigPlanet = cameraState.index === -1;

      let cameraTargetPosition;
      let lookAtTarget;

      if (isBigPlanet) {
        cameraTargetPosition = isMobileRef.current
            ? { x: -1.5, y: 1.2, z: 1.8 }
            : { x: -1.8, y: 1.5, z: 2.2 };
        lookAtTarget = { x: 0, y: -0.5 + 0.8, z: 0 };
      } else {
        const dir = cameraState.direction;
        const distance = isMobileRef.current ? 2.8 : 3.2;

        cameraTargetPosition = {
          x: target.x - dir.x * distance,
          y: target.y - dir.y * distance + 0.8,
          z: target.z - dir.z * distance
        };
        lookAtTarget = { x: target.x, y: target.y + 0.5, z: target.z };
      }

      cameraAnimationRef.current = gsap.timeline()
          .to(cameraPositionRef.current, {
            ...cameraTargetPosition,
            duration: 2.5,
            ease: 'power2.inOut',
            onUpdate: updateCamera, // Только обновление камеры, без рендера
          })
          .to(lookAtTargetRef.current, {
            ...lookAtTarget,
            duration: 2.5,
            ease: 'power2.inOut',
            onUpdate: updateCamera, // Только обновление камеры, без рендера
          }, 0);
    }
  }, [cameraState]);

  // Camera return to initial view
  useEffect(() => {
    if (cameraState.mode === 'initial' && cameraRef.current && sceneRef.current) {
      if (cameraAnimationRef.current) {
        cameraAnimationRef.current.kill();
      }

      const initialPosition = isMobileRef.current
          ? { x: 0, y: 6.5, z: 10 }
          : { x: 0, y: 5, z: 7 };

      cameraAnimationRef.current = gsap.timeline()
          .to(cameraPositionRef.current, {
            ...initialPosition,
            duration: 2.5,
            ease: 'power2.inOut',
            onUpdate: updateCamera, // Только обновление камеры, без рендера
          })
          .to(cameraRotationRef.current, {
            x: 0,
            y: 0,
            z: THREE.MathUtils.degToRad(-30),
            duration: 2.5,
            ease: 'power2.inOut',
            onUpdate: updateCamera, // Только обновление камеры, без рендера
          }, 0)
          .to(lookAtTargetRef.current, {
            x: 0,
            y: -0.5,
            z: 0,
            duration: 2.5,
            ease: 'power2.inOut',
            onUpdate: updateCamera, // Только обновление камеры, без рендера
            onComplete: () => {
              if (observerRef.current) {
                observerRef.current.observe(mountRef.current);
              }
            },
          }, 0);
    }
  }, [cameraState]);

  const handleBack = () => {
    setCameraState({ mode: 'initial' });
  };

  // Обработчик для предотвращения проблем с hot reload
  useEffect(() => {
    if (process.env.NODE_ENV === 'development') {
      const handleKeyDown = (event) => {
        // Ctrl+R или F5 - принудительная перезагрузка
        if ((event.ctrlKey && event.key === 'r') || event.key === 'F5') {
          console.log('🔄 Принудительная перезагрузка страницы');
          cleanupScene();
        }
      };

      window.addEventListener('keydown', handleKeyDown);

      return () => {
        window.removeEventListener('keydown', handleKeyDown);
      };
    }
  }, []);

  return (
      <div className="w-full h-[100dvh] bg-gradient-to-b from-[#0a0f2b] via-[#050716] to-[#000000] relative overflow-hidden">
        {isLoading && (
            <div style={loaderStyle}>
              <div className="relative flex flex-col items-center">
                <div className="relative w-8 h-8">
                  <div className="absolute inset-0 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 animate-spin-slow">
                    <div className="absolute inset-0 rounded-full bg-gradient-to-br from-blue-300/50 to-purple-400/50 animate-pulse"></div>
                  </div>
                  <div className="absolute w-12 h-12 animate-orbit" style={{ animationDuration: '3s' }}>
                    <div className="absolute w-4 h-4 bg-white rounded-full top-0 left-1/2 -translate-x-1/2"></div>
                  </div>
                  <div className="absolute w-18 h-18 animate-orbit-reverse" style={{ animationDuration: '4s' }}>
                    <div className="absolute w-3 h-3 bg-blue-300 rounded-full top-0 left-1/2 -translate-x-1/2"></div>
                  </div>
                  <div className="absolute w-21 h-21 animate-orbit" style={{ animationDuration: '5s' }}>
                    <div className="absolute w-2 h-2 bg-purple-300 rounded-full top-0 left-1/2 -translate-x-1/2"></div>
                  </div>
                </div>
                <p className="mt-4 text-white text-lg font-semibold animate-pulse z-10000">
                  Загрузка космоса...
                </p>
              </div>
            </div>
        )}
        <Header
            onBack={handleBack}
            showChatLabel={cameraState.mode === 'focused' && cameraState.index === -1}
            showBackButton={cameraState.mode === 'focused'}
            showStats={showStats}
            onToggleStats={() => setShowStats(!showStats)}
            isMobile={isMobile}
        />
        <EmotionalStats
            isVisible={showStats}
            isMobile={isMobile}
            onClose={() => setShowStats(false)}
            focusOnPlanet={focusOnPlanet}
        />
        {cameraState.mode !== 'focused' && (
         <BottomTab
            focusOnPlanet={focusOnPlanet}
            randomEmotion={randomEmotion}
          />
        )}
        <div ref={mountRef} className="w-full h-full absolute top-0 left-0 z-0" />
        {cameraState.mode === 'focused' && cameraState.index === -1 && (
            <div className="absolute top-0 left-0 w-full h-full z-10 pointer-events-auto">
              <Chat onBack={handleBack} />
            </div>
        )}
        {cameraState.mode === 'focused' && cameraState.index !== -1 && (
            <PlanetInterface
                planetIndex={cameraState.index}
                planetData={{ name: cameraState.emotion }}
                onClose={handleBack}
            />
        )}
      </div>
  );
}
