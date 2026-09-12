// planetLoader.js (обновленный с мобильным и десктопным скейлом из конфига)
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { getCachedModel, cacheModel } from './idbCache';

// Конфигурация планет с отдельными скейлами для мобильных и десктопных
export const planetConfigs = [
    {
        name: 'Центральная',
        emotion: 'user',
        modelPath: '/models/Main V2.glb',
        ringsPath: '/models/Rings FOR MAIN V2.glb',
        cloudsPath: null,
        radius: 0,
        scale: 0.8,      // Десктопный скейл
        mobileScale: 0.7, // Мобильный скейл
        isCentral: true,
        rotationSpeed: 0.02,
        rotationDirection: 1, // 1: против часовой, -1: по часовой
        ringsRotationSpeed: 0.008,
        ringsRotationDirection: 1
    },
    {
        name: 'Радость',
        emotion: 'joy',
        modelPath: '/models/Happyv2.glb',
        cloudsPath: null,
        radius: 1.8,
        angle: 0,
        scale: 0.4,      // Десктопный скейл
        mobileScale: 0.35, // Мобильный скейл
        rotationSpeed: 0.02,
        rotationDirection: 1
    },
    {
        name: 'Доверие',
        emotion: 'trust',
        modelPath: '/models/Trust.glb',
        cloudsPath: null,
        radius: 2.0,
        angle: 45,
        scale: 0.35,     // Десктопный скейл
        mobileScale: 0.3,  // Мобильный скейл
        rotationSpeed: 0.02,
        rotationDirection: -1
    },
    {
        name: 'Страх',
        emotion: 'fear',
        modelPath: '/models/Fear.glb',
        cloudsPath: null,
        radius: 2.9,
        angle: 75,
        scale: 0.25,     // Десктопный скейл
        mobileScale: 0.2,  // Мобильный скейл
        rotationSpeed: 0.02,
        rotationDirection: 1
    },
    {
        name: 'Удивление',
        emotion: 'surprise',
        modelPath: '/models/Surpv2.glb',
        cloudsPath: null,
        radius: 1.9,
        angle: 120,
        scale: 0.25,     // Десктопный скейл
        mobileScale: 0.2,  // Мобильный скейл
        rotationSpeed: 0.02,
        rotationDirection: -1
    },
    {
        name: 'Грусть',
        emotion: 'sadness',
        modelPath: '/models/SadV2.glb',
        iconPath: '/models/SadALONE.glb',
        cloudsPath: null,
        radius: 2.6,
        angle: 190,
        scale: 0.3,      // Десктопный скейл (было 0.2)
        mobileScale: 0.4,  // Мобильный скейл (было 0.4)
        rotationSpeed: 0.02,
        rotationDirection: 1
    },
    {
        name: 'Отвращение',
        emotion: 'disgust',
        modelPath: '/models/disgustV2.glb',
        cloudsPath: null,
        radius: 1.5,
        angle: 175,
        scale: 0.3,      // ✅ ИСПРАВЛЕНО: увеличен с 0.2 до 0.3 для десктопа
        mobileScale: 0.25, // ✅ ИСПРАВЛЕНО: 0.25 для мобильных
        rotationSpeed: 0.02,
        rotationDirection: -1
    },
    {
        name: 'Гнев',
        emotion: 'anger',
        modelPath: '/models/AngrV2.glb',
        cloudsPath: null,
        radius: 2.0,
        angle: 220,
        scale: 0.4,      // Десктопный скейл
        mobileScale: 0.35, // Мобильный скейл
        rotationSpeed: 0.02,
        rotationDirection: 1
    },
    {
        name: 'Ожидание',
        emotion: 'anticipation',
        modelPath: '/models/Wait.glb',
        cloudsPath: null,
        radius: 2.5,
        angle: 315,
        scale: 0.4,      // Десктопный скейл
        mobileScale: 0.35, // Мобильный скейл
        rotationSpeed: 0.02,
        rotationDirection: -1
    }
];

// Функция загрузки одной модели с кэшированием (как blob)
async function loadOnePlanet(loader, planet) {
    const { modelPath, cloudsPath, ringsPath, name } = planet;
    console.log(`⏳ Начало загрузки для ${name} (${modelPath})`);

    const loadModel = async (path) => {
        console.log(`🔍 Проверка кэша для ${path}`);
        const cached = await getCachedModel(path);
        if (cached) {
            console.log(`📂 Используем кэш для ${path}`);
            const blobUrl = URL.createObjectURL(cached);
            try {
                return await new Promise((resolve, reject) => {
                    loader.load(blobUrl, resolve, undefined, reject);
                });
            } finally {
                URL.revokeObjectURL(blobUrl);
            }
        }

        console.log(`🌐 Загрузка с сервера: ${path}`);
        try {
            const response = await fetch(path);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const blob = await response.blob();
            await cacheModel(path, blob);
            console.log(`💾 Закэшировано: ${path}`);
            const blobUrl = URL.createObjectURL(blob);
            try {
                return await new Promise((resolve, reject) => {
                    loader.load(blobUrl, resolve, undefined, reject);
                });
            } finally {
                URL.revokeObjectURL(blobUrl);
            }
        } catch (error) {
            console.warn(`Не удалось загрузить ${path}, используется заглушка`);
            throw error;
        }
    };

    try {
        const planetModel = await loadModel(modelPath);

        let cloudsModel = null;
        if (cloudsPath) {
            try {
                cloudsModel = await loadModel(cloudsPath);
            } catch (error) {
                console.warn(`Облака для ${name} не загружены:`, error);
            }
        }

        let ringsModel = null;
        if (ringsPath) {
            try {
                ringsModel = await loadModel(ringsPath);
            } catch (error) {
                console.error(`Кольца для ${name} не загружены:`, error);
            }
        }

        console.log(`✅ Успешно загружена модель для ${name}`);
        return {
            planet: planetModel.scene,
            clouds: cloudsModel ? cloudsModel.scene : null,
            rings: ringsModel ? ringsModel.scene : null
        };
    } catch (error) {
        console.error(`Ошибка загрузки модели для ${name}:`, error);
        const geometry = new THREE.SphereGeometry(0.5, 32, 32);
        const material = new THREE.MeshBasicMaterial({
            color: 0x888888,
            transparent: true,
            opacity: 0.8
        });
        console.log(`🛠️ Создана заглушка для ${name}`);
        return {
            planet: new THREE.Mesh(geometry, material),
            clouds: null,
            rings: null
        };
    }
}

// Функция загрузки моделей
export const loadPlanetModels = async (isMobile) => {
    console.log('🚀 Начало загрузки всех моделей планет');
    const loader = new GLTFLoader();
    const dracoLoader = new DRACOLoader();

    try {
        dracoLoader.setDecoderPath('draco/');
        loader.setDRACOLoader(dracoLoader);
        console.log('✅ Draco loader настроен');
    } catch (error) {
        console.warn('⚠️ Draco loader не доступен, продолжаем без него');
        loader.setDRACOLoader(null);
    }

    const loadedModels = {};
    for (const planet of planetConfigs) {
        try {
            loadedModels[planet.emotion] = await loadOnePlanet(loader, planet);
            console.log(`✅ Завершена загрузка для ${planet.name}`);
        } catch (error) {
            console.error(`Не удалось загрузить ${planet.name}, используется заглушка`);
            loadedModels[planet.emotion] = {
                planet: new THREE.Mesh(
                    new THREE.SphereGeometry(0.5, 32, 32),
                    new THREE.MeshBasicMaterial({ color: 0x888888, transparent: true, opacity: 0.8 })
                ),
                clouds: null,
                rings: null
            };
        }
    }

    console.log('✅ Все модели планет загружены или заменены заглушками');
    return loadedModels;
};

// Функция создания планет в сцене (ПОЛНОСТЬЮ ПЕРЕПИСАНА с использованием конфига)
export const createPlanetsInScene = (scene, planetModels, isMobile) => {
    const planets = [];

    console.log(`📱 Режим: ${isMobile ? 'мобильный' : 'десктопный'}`);

    planetConfigs.forEach((planet, index) => {
        if (planet.isCentral) {
            // Центральная планета
            const centralPlanet = planetModels['user'].planet.clone();

            // ✅ Берем скейл из конфига в зависимости от устройства
            const scale = isMobile ? planet.mobileScale : planet.scale;
            console.log(`🌍 Центральная планета: scale=${scale} (${isMobile ? 'mobile' : 'desktop'})`);

            centralPlanet.scale.set(scale, scale, scale);
            centralPlanet.position.set(0, -0.5, 0);

            centralPlanet.userData = {
                isBigPlanet: true,
                index: -1,
                target: { x: 0, y: -0.5, z: 0 },
                originalPosition: { x: 0, y: -0.5, z: 0 },
                emotion: 'user',
                rotationSpeed: planet.rotationSpeed,
                rotationDirection: planet.rotationDirection
            };

            centralPlanet.userData.update = (delta) => {
                centralPlanet.rotation.y += centralPlanet.userData.rotationSpeed * centralPlanet.userData.rotationDirection * delta;
            };

            setupModelShadows(centralPlanet);
            scene.add(centralPlanet);
            planets.push(centralPlanet);

            // Облака для центральной планеты
            if (planetModels['user'].clouds) {
                const centralClouds = planetModels['user'].clouds.clone();
                centralClouds.scale.set(scale, scale, scale);
                centralClouds.position.set(0, -0.5, 0);
                centralClouds.userData = { ...centralPlanet.userData };
                centralClouds.userData.update = (delta) => {
                    centralClouds.rotation.y += centralClouds.userData.rotationSpeed * centralClouds.userData.rotationDirection * delta;
                };
                setupModelShadows(centralClouds);
                scene.add(centralClouds);
                planets.push(centralClouds);
            }

            // КОЛЬЦА ДЛЯ ЦЕНТРАЛЬНОЙ ПЛАНЕТЫ - НИЖЕ ТУМАННОСТЕЙ!
            if (planetModels['user'].rings) {
                const centralRings = planetModels['user'].rings.clone();

                // Масштабируем кольца относительно скейла центральной планеты
                const ringsScale = isMobile ? 0.25 : 0.3; // Меньше для мобильных
                centralRings.scale.set(ringsScale, ringsScale, ringsScale);

                // КОЛЬЦА НИЖЕ ТУМАННОСТЕЙ ПО Y
                centralRings.position.set(0, -0.55, 0);

                // ОРИЕНТАЦИЯ КОЛЕЦ (горизонтально)
                centralRings.rotation.x = 0;

                // ПРИОРИТЕТ РЕНДЕРА: КОЛЬЦА РЕНДЕРЯТСЯ ПЕРВЫМИ
                centralRings.renderOrder = 0;
                console.log(`📊 Кольца renderOrder=0 (туманности поверх с renderOrder=1)`);

                // ПРОВЕРКА РАДИУСА КОЛЕЦ
                let maxRadius = 0;
                centralRings.traverse((child) => {
                    if (child.isMesh && child.geometry) {
                        try {
                            child.geometry.computeBoundingSphere();
                            if (child.geometry.boundingSphere) {
                                const localRadius = child.geometry.boundingSphere.radius;
                                const modelRadius = localRadius * ringsScale;
                                if (modelRadius > maxRadius) {
                                    maxRadius = modelRadius;
                                }
                            }
                        } catch (error) {
                            // Игнорируем ошибки
                        }
                    }
                });

                console.log(`📏 Центральные кольца: scale=${ringsScale}, итоговый радиус = ${(maxRadius).toFixed(2)} (лимит: ~1.5)`);
                console.log(`🎯 Кольца расположены ниже туманностей: y = -0.55`);

                // УЛУЧШЕНИЕ ВИДИМОСТИ КОЛЕЦ + УДАЛЕНИЕ ЧЕРНОГО ФОНА
                centralRings.traverse((child) => {
                    if (child.isMesh && child.material) {
                        const materials = Array.isArray(child.material) ? child.material : [child.material];
                        materials.forEach((material) => {
                            // УДАЛЕНИЕ ЧЕРНОГО ФОНА: ПЕРЕКЛЮЧАЕМ НА ADDITIVE BLENDING
                            material.blending = THREE.AdditiveBlending;
                            material.transparent = true;
                            material.depthWrite = false;
                            material.depthTest = true;
                            material.side = THREE.DoubleSide;
                            material.needsUpdate = true;

                            // Увеличиваем яркость
                            if (material.color) {
                                material.color.multiplyScalar(1.5);
                            }
                            if (material.opacity !== undefined) {
                                material.opacity = Math.min(1.0, material.opacity * 1.8);
                            }

                            // Настройки для свечения
                            if (material.isMeshStandardMaterial || material.isMeshPhysicalMaterial) {
                                material.emissive = new THREE.Color(0x444444);
                                material.emissiveIntensity = 0.3;
                                material.roughness = 0.1;
                                material.metalness = 0.8;
                            }
                        });
                    }
                });

                console.log(`🖼️ Черный фон колец удален: используем AdditiveBlending с альфа-каналом`);

                centralRings.userData = {};
                centralRings.userData.rotationSpeed = planet.ringsRotationSpeed;
                centralRings.userData.rotationDirection = planet.ringsRotationDirection;
                centralRings.userData.update = (delta) => {
                    centralRings.rotation.y += centralRings.userData.rotationSpeed * centralRings.userData.rotationDirection * delta;
                };
                setupModelShadows(centralRings);
                scene.add(centralRings);
                // Удаляем push в planets, чтобы raycaster игнорировал кольца и не вызывал фокус на центральную планету при клике в области перекрытия
                // planets.push(centralRings);
            }

        } else {
            // Орбитальные планеты
            const angle = THREE.MathUtils.degToRad(planet.angle);
            const position = {
                x: Math.cos(angle) * planet.radius,
                y: -0.5,
                z: Math.sin(angle) * planet.radius
            };

            // ✅ Берем скейл из конфига в зависимости от устройства
            const scale = isMobile ? planet.mobileScale : planet.scale;
            console.log(`🌍 ${planet.name} (${planet.emotion}): scale=${scale} (${isMobile ? 'mobile' : 'desktop'})`);

            // Планета
            const orbitalPlanet = planetModels[planet.emotion].planet.clone();
            orbitalPlanet.scale.set(scale, scale, scale);
            orbitalPlanet.position.set(position.x, position.y, position.z);

            orbitalPlanet.userData = {
                isBigPlanet: false,
                index: index - 1, // index начинается с 0 для орбитальных планет
                target: position,
                originalPosition: { ...position },
                emotion: planet.emotion,
                name: planet.name,
                rotationSpeed: planet.rotationSpeed,
                rotationDirection: planet.rotationDirection,
                currentAngle: angle,
                radius: planet.radius
            };

            orbitalPlanet.userData.update = (delta) => {
                orbitalPlanet.rotation.y += orbitalPlanet.userData.rotationSpeed * orbitalPlanet.userData.rotationDirection * delta;
            };

            setupModelShadows(orbitalPlanet);
            scene.add(orbitalPlanet);
            planets.push(orbitalPlanet);

            // Облака для орбитальной планеты
            if (planetModels[planet.emotion].clouds) {
                const orbitalClouds = planetModels[planet.emotion].clouds.clone();
                orbitalClouds.scale.set(scale, scale, scale); // Тот же скейл что и у планеты
                orbitalClouds.position.set(position.x, position.y, position.z);
                orbitalClouds.userData = { ...orbitalPlanet.userData };
                orbitalClouds.userData.update = (delta) => {
                    orbitalClouds.rotation.y += orbitalClouds.userData.rotationSpeed * orbitalClouds.userData.rotationDirection * delta;
                };
                setupModelShadows(orbitalClouds);
                scene.add(orbitalClouds);
                planets.push(orbitalClouds);
            }

            // Орбитальное кольцо (тонкое)
            const ringGeometry = new THREE.RingGeometry(planet.radius - 0.01, planet.radius + 0.01, 64);
            const ringMaterial = new THREE.MeshBasicMaterial({
                color: 0xffffff,
                side: THREE.DoubleSide,
                transparent: true,
                opacity: 0.03,
            });
            const ring = new THREE.Mesh(ringGeometry, ringMaterial);
            ring.rotation.x = Math.PI / 2;
            ring.position.y = -0.5;
            ring.renderOrder = 2;
            scene.add(ring);
        }
    });

    console.log('✅ Все планеты созданы с правильными масштабами из конфига');
    return planets;
};

// Вспомогательная функция для настройки теней (без изменений)
const setupModelShadows = (model) => {
    model.traverse((child) => {
        if (child.isMesh) {
            child.castShadow = true;
            child.receiveShadow = true;
        }
    });
};

// Функция получения позиций планет (без изменений)
export const getPlanetPositions = () => {
    return planetConfigs
        .filter(planet => !planet.isCentral)
        .map(planet => {
            const angle = THREE.MathUtils.degToRad(planet.angle);
            return {
                x: Math.cos(angle) * planet.radius,
                y: -0.5,
                z: Math.sin(angle) * planet.radius
            };
        });
};

// ✅ НОВАЯ ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: для получения текущих масштабов планет
export const getPlanetScales = (isMobile) => {
    return planetConfigs.map(planet => ({
        name: planet.name,
        emotion: planet.emotion,
        scale: isMobile ? planet.mobileScale : planet.scale,
        isMobile
    }));
};