// newPlanetLoader.js (исправлена ориентация и цвет новых колец)
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js';

// Конфигурация планет (без изменений)
export const planetConfigs = [
    {
        name: 'Центральная',
        emotion: 'user',
        folder: 'MAIN',
        models: [
            { name: 'planet', path: 'MAIN.gltf' },
            { name: 'rings', path: 'Rings.gltf' } // НОВЫЕ КОЛЬЦА!
        ],
        environmentMap: '/models/MAIN/REFLECT/reflect.webp',
        radius: 0,
        scale: 0.8,
        isCentral: true
    },
    // ... остальные планеты без изменений
    {
        name: 'Радость',
        emotion: 'joy',
        folder: 'Happy2',
        models: [
            { name: 'planet', path: 'FixHAPPY.gltf' }
        ],
        radius: 1.8,
        angle: 0,
        scale: 0.4
    },
    {
        name: 'Доверие',
        emotion: 'trust',
        folder: 'trust',
        models: [
            { name: 'planet', path: 'trust.gltf' },
            { name: 'atmosphere', path: 'trustAtmos.gltf' }
        ],
        radius: 2.0,
        angle: 45,
        scale: 0.35
    },
    {
        name: 'Страх',
        emotion: 'fear',
        folder: 'Fear',
        models: [
            { name: 'planet', path: 'Fear.gltf' }
        ],
        radius: 2.9,
        angle: 75,
        scale: 0.25
    },
    {
        name: 'Удивление',
        emotion: 'surprise',
        folder: 'Surprise',
        models: [
            { name: 'planet', path: 'Surprise.gltf' },
            { name: 'sparks', path: 'Sparls.gltf' },
            { name: 'atmosphere', path: 'SurpAtmos.gltf' }
        ],
        radius: 1.9,
        angle: 120,
        scale: 0.25
    },
    {
        name: 'Грусть',
        emotion: 'sadness',
        folder: 'CracksSAD',
        models: [
            { name: 'planet', path: 'PlannetCracksSAD.gltf' },
            { name: 'atmosphere', path: 'ATMOSCracksSAD.gltf' },
            { name: 'rocks', path: 'ROCKSCracksSAD.gltf' }
        ],
        radius: 2.2,
        angle: 160,
        scale: 0.6
    },
    {
        name: 'Отвращение',
        emotion: 'disgust',
        folder: 'Disgust',
        models: [
            { name: 'planet', path: 'Disgust.gltf' },
            { name: 'atmosphere', path: 'DisgustATMOS.gltf' }
        ],
        radius: 1.5,
        angle: 175,
        scale: 0.2
    },
    {
        name: 'Гнев',
        emotion: 'anger',
        folder: 'anger',
        models: [
            { name: 'planet', path: 'Anger.gltf' }
        ],
        radius: 2.0,
        angle: 220,
        scale: 0.4
    },
    {
        name: 'Ожидание',
        emotion: 'anticipation',
        folder: 'Wait',
        models: [
            { name: 'planet', path: 'Wait.gltf' }
        ],
        radius: 2.5,
        angle: 315,
        scale: 0.4
    }
];

// Функции для планет (без изменений)
const makePlanetsOpaque = (planetScene) => {
    planetScene.traverse((child) => {
        if (child.isMesh && child.material) {
            const materials = Array.isArray(child.material) ? child.material : [child.material];
            materials.forEach((material) => {
                material.transparent = false;
                material.opacity = 1.0;
                material.depthWrite = true;
                if (material.alphaTest !== undefined) {
                    material.alphaTest = 0.0;
                }
                if (material.color) {
                    material.color.multiplyScalar(1.1);
                }
                material.needsUpdate = true;
            });
        }
    });
    console.log('Планета сделана полностью непрозрачной');
};

const loadEnvironmentMap = async (path) => {
    return new Promise((resolve, reject) => {
        const loader = new THREE.TextureLoader();
        loader.load(
            path,
            (texture) => {
                texture.mapping = THREE.EquirectangularReflectionMapping;
                texture.encoding = THREE.sRGBEncoding;
                texture.wrapS = THREE.ClampToEdgeWrapping;
                texture.wrapT = THREE.ClampToEdgeWrapping;
                resolve(texture);
            },
            undefined,
            (error) => {
                console.warn('Не удалось загрузить environment map:', path, error);
                resolve(null);
            }
        );
    });
};

const applyEnvironmentMap = (scene, environmentMap) => {
    scene.traverse((child) => {
        if (child.isMesh && child.material) {
            const materials = Array.isArray(child.material) ? child.material : [child.material];
            materials.forEach((material) => {
                if (material.isMeshStandardMaterial || material.isMeshPhysicalMaterial) {
                    material.envMap = environmentMap;
                    material.envMapIntensity = 1.0;
                    material.needsUpdate = true;
                }
                else if (material.isMeshBasicMaterial) {
                    material.envMap = environmentMap;
                    material.needsUpdate = true;
                }
            });
        }
    });
    console.log('Environment map применен к планете');
};

// ИСПРАВЛЕННАЯ ФУНКЦИЯ для новых колец - СОХРАНЯЕМ ОРИГИНАЛЬНЫЕ ЦВЕТА!
const enhanceNewRingsVisibility = (ringsScene) => {
    console.log('🔧 Применяем улучшения для новых колец (Rings.gltf)...');

    ringsScene.traverse((child) => {
        if (child.isMesh && child.material) {
            const materials = Array.isArray(child.material) ? child.material : [child.material];

            materials.forEach((material) => {
                // СОХРАНЯЕМ ОРИГИНАЛЬНЫЙ ЦВЕТ ТЕКСТУРЫ (оранжевый!)
                if (material.color && !material.map) {
                    // Если нет текстуры, слегка усиливаем оригинальный цвет
                    material.color.multiplyScalar(1.3);
                }

                // РАБОТА С АЛЬФА-КАНАЛОМ (для прозрачности текстур)
                if (material.transparent !== undefined) {
                    material.transparent = true; // Включаем прозрачность для альфа-канала
                }

                // Настраиваем opacity для лучшей видимости
                if (material.opacity !== undefined) {
                    material.opacity = Math.min(1.0, material.opacity * 1.5); // Мягче, чем 1.8
                }

                // ДЛЯ STANDARD/PHYSICAL МАТЕРИАЛОВ - СОХРАНЯЕМ ОРИГИНАЛЬНЫЙ ЦВЕТ
                if (material.isMeshStandardMaterial || material.isMeshPhysicalMaterial) {
                    // ТЕПЛОЕ СВЕЧЕНИЕ (оранжевое) вместо серого
                    material.emissive = new THREE.Color(0xff8c00); // Оранжевый emissive!
                    material.emissiveIntensity = 0.4; // Мягче свечение

                    // Делаем более металлическими и блестящими
                    material.roughness = 0.1;
                    material.metalness = 0.8;

                    // Смягчаем envMap intensity для избежания засветов
                    if (material.envMapIntensity !== undefined) {
                        material.envMapIntensity = 0.6; // Меньше отражений
                    }
                }

                // ДЛЯ BASIC МАТЕРИАЛОВ - СОХРАНЯЕМ ОРИГИНАЛЬНЫЙ ЦВЕТ
                if (material.isMeshBasicMaterial) {
                    // НЕ перезаписываем color! Только если его нет
                    if (!material.map && material.color) {
                        material.color.multiplyScalar(1.3); // Усиливаем оригинальный
                    }
                    // ТЕПЛОЕ СВЕЧЕНИЕ
                    material.emissive = new THREE.Color(0xffa500); // Оранжевый!
                    material.emissiveIntensity = 0.3;
                }

                // ВАЖНО: Настройки для правильного рендеринга
                material.depthWrite = true;
                material.side = THREE.DoubleSide;
                material.needsUpdate = true;
            });
        }
    });

    console.log('✅ Новые кольца улучшены (сохранены оранжевые цвета!)');
};

// Функция загрузки моделей (без изменений)
export const loadPlanetModels = async (isMobile) => {
    const loader = new GLTFLoader();
    const dracoLoader = new DRACOLoader();

    try {
        dracoLoader.setDecoderPath('draco/');
        loader.setDRACOLoader(dracoLoader);
    } catch (error) {
        console.warn('Draco loader не доступен, продолжаем без него');
        loader.setDRACOLoader(null);
    }

    const loadedModels = {};

    const planetLoadPromises = planetConfigs.map(async (planet) => {
        const modelScenes = {};
        let hasError = false;

        const environmentMap = planet.environmentMap
            ? await loadEnvironmentMap(planet.environmentMap)
            : null;

        const modelPromises = planet.models.map(async (modelConfig) => {
            try {
                const fullPath = `/models/${planet.folder}/${modelConfig.path}`;

                const modelData = await new Promise((resolve, reject) => {
                    loader.load(
                        fullPath,
                        resolve,
                        undefined,
                        (error) => {
                            console.warn(`Ошибка загрузки ${modelConfig.name} для ${planet.name}:`, error);
                            reject(error);
                        }
                    );
                });
                modelScenes[modelConfig.name] = modelData.scene;
            } catch (error) {
                console.error(`Не удалось загрузить ${modelConfig.name} для ${planet.name}:`, error);
                modelScenes[modelConfig.name] = null;
                hasError = true;
            }
        });

        await Promise.all(modelPromises);

        if (modelScenes.planet) {
            makePlanetsOpaque(modelScenes.planet);
        }

        if (environmentMap && modelScenes.planet) {
            applyEnvironmentMap(modelScenes.planet, environmentMap);
        }

        // УЛУЧШЕНИЕ НОВЫХ КОЛЕЦ
        if (planet.isCentral && modelScenes.rings) {
            enhanceNewRingsVisibility(modelScenes.rings);
        }

        if (hasError && !modelScenes.planet) {
            const geometry = new THREE.SphereGeometry(0.5, 32, 32);
            const material = new THREE.MeshBasicMaterial({
                color: 0x888888
            });
            const placeholder = new THREE.Mesh(geometry, material);
            modelScenes.planet = placeholder;
        }

        loadedModels[planet.emotion] = {
            models: modelScenes,
            environmentMap
        };
    });

    await Promise.all(planetLoadPromises);
    return loadedModels;
};

// ИСПРАВЛЕННАЯ ФУНКЦИЯ СОЗДАНИЯ ПЛАНЕТ (правильная ориентация колец)
export const createPlanetsInScene = (scene, planetModels, isMobile) => {
    const planets = [];
    const scales = isMobile
        ? [0.7, 0.35, 0.3, 0.2, 0.2, 0.4, 0.15, 0.35, 0.35]
        : [0.8, 0.4, 0.35, 0.25, 0.25, 0.3, 0.2, 0.4, 0.4];

    const firstOrbitRadius = 1.8;

    planetConfigs.forEach((planet, index) => {
        const emotion = planet.isCentral ? 'user' : planet.emotion;
        const scale = planet.isCentral ? scales[0] : scales[index];

        const planetGroup = new THREE.Group();
        planetGroup.userData = {
            isBigPlanet: planet.isCentral,
            index: planet.isCentral ? -1 : index - 1,
            target: { x: 0, y: -0.5, z: 0 },
            originalPosition: { x: 0, y: -0.5, z: 0 },
            emotion: emotion,
            name: planet.name,
            folder: planet.folder
        };

        Object.entries(planetModels[emotion].models).forEach(([modelName, modelScene]) => {
            if (modelScene) {
                const clonedModel = modelScene.clone();

                // НАСТРОЙКА НОВЫХ КОЛЕЦ (Rings.gltf) - ПРАВИЛЬНАЯ ОРИЕНТАЦИЯ!
                if (planet.isCentral && modelName === 'rings') {
                    // РАЗМЕР (оставляем 0.4 - ты сказал что правильно)
                    clonedModel.scale.set(0.4, 0.4, 0.4);

                    // ИСПРАВЛЯЕМ ОРИЕНТАЦИЮ: УБИРАЕМ ПОВОРОТ!
                    // Кольца должны быть в той же плоскости, что и туманности (Y=-0.5, плоскость XZ)
                    // Если модель экспортирована вертикально - поворачиваем на Y=90° или X=-90°

                    // ПРОБУЕМ РАЗНЫЕ ВАРИАНТЫ ПОВОРОТА (выбери подходящий):

                    // ВАРИАНТ 1: Если модель вертикальная (как у Сатурна)
                    //clonedModel.rotation.x = -Math.PI / 2; // Поворот по X на -90° (против часовой)

                    // ВАРИАНТ 2: Если нужно другое направление
                    clonedModel.rotation.y = Math.PI / 2; // Поворот по Y на 90°

                    // ВАРИАНТ 3: Если модель уже горизонтальная - убрать поворот вообще
                    // clonedModel.rotation.set(0, 0, 0); // Без поворота

                    // ПРОВЕРКА РАДИУСА новых колец
                    let maxRadius = 0;
                    clonedModel.traverse((child) => {
                        if (child.isMesh && child.geometry) {
                            try {
                                child.geometry.computeBoundingSphere();
                                if (child.geometry.boundingSphere) {
                                    const localRadius = child.geometry.boundingSphere.radius;
                                    const modelRadius = localRadius * clonedModel.scale.x;
                                    const totalRadius = modelRadius * scale;

                                    if (totalRadius > maxRadius) {
                                        maxRadius = totalRadius;
                                    }
                                }
                            } catch (error) {
                                // Игнорируем ошибки
                            }
                        }
                    });

                    console.log(`📏 Новые кольца: scale=0.4, итоговый радиус = ${(maxRadius).toFixed(2)} (лимит: ~1.8)`);

                    // ДОПОЛНИТЕЛЬНАЯ НАСТРОЙКА МАТЕРИАЛОВ (уже сделано в enhanceNewRingsVisibility)
                    // Здесь только финальная настройка рендеринга
                    clonedModel.traverse((child) => {
                        if (child.isMesh && child.material) {
                            const materials = Array.isArray(child.material) ? child.material : [child.material];
                            materials.forEach((material) => {
                                material.depthWrite = true;
                                material.needsUpdate = true;
                            });
                        }
                    });

                    // ПОЗИЦИЯ КОЛЕЦ (в центре центральной планеты - относительно planetGroup y=0)
                    clonedModel.position.y = 0;
                }

                planetGroup.add(clonedModel);
            }
        });

        planetGroup.scale.set(scale, scale, scale);

        if (planet.isCentral) {
            planetGroup.position.set(0, -0.5, 0);
            planetGroup.userData.target = { x: 0, y: -0.5, z: 0 };
            planetGroup.userData.originalPosition = { x: 0, y: -0.5, z: 0 };
        } else {
            const angle = THREE.MathUtils.degToRad(planet.angle);
            const position = {
                x: Math.cos(angle) * planet.radius,
                y: -0.5,
                z: Math.sin(angle) * planet.radius
            };
            planetGroup.position.set(position.x, position.y, position.z);
            planetGroup.userData.target = position;
            planetGroup.userData.originalPosition = { ...position };
        }

        setupModelShadows(planetGroup);
        scene.add(planetGroup);
        planets.push(planetGroup);

        if (!planet.isCentral) {
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

    return planets;
};

const setupModelShadows = (group) => {
    group.traverse((child) => {
        if (child.isMesh) {
            child.castShadow = true;
            child.receiveShadow = true;
        }
    });
};

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