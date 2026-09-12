import * as THREE from 'three';

// КОНФИГУРАЦИЯ СТАТИЧНЫХ ДУГОВИДНЫХ ТУМАННОСТЕЙ
export const nebulaConfigs = [
    {
        name: 'nebula_01',
        colorPath: '/textures/nebulas/nebula_02_color.png',
        alphaPath: '/textures/nebulas/nebula_02_alpha.png',
        emissionPath: '/textures/nebulas/nebula_02_emission.png',
        arcAngle: 180,
        arcRadius: 2.0,
        arcWidth: 2.0,
        arcHeight: 0.6,
        startAngle: 20,
        mirrorTexture: true,
        opacity: 0.65,
        displacementScale: 0.3,
        noiseScale: 0.7,
        noiseAmplitude: 0.15,
        flowSpeed: 0.01,
        flowDirection: { x: -0.01, y: 0.002 },
        colorModAmplitude: 0.08
    },
    {
        name: 'nebula_02',
        colorPath: '/textures/nebulas/nebula_02_color.png',
        alphaPath: '/textures/nebulas/nebula_02_alpha.png',
        emissionPath: '/textures/nebulas/nebula_02_emission.png',
        arcAngle: 210,
        arcRadius: 2.1,
        arcWidth: 2.0,
        arcHeight: 0.6,
        startAngle: 180,
        mirrorTexture: false,
        opacity: 0.65,
        displacementScale: 0.3,
        noiseScale: 0.7,
        noiseAmplitude: 0.15,
        flowSpeed: 0.01,
        flowDirection: { x: 0.01, y: 0.002 },
        colorModAmplitude: 0.08
    }
];

// ВЕРШИННЫЙ ШЕЙДЕР
const vertexShader = `
  uniform float uArcAngle;
  uniform float uArcRadius;
  uniform float uArcWidth;
  uniform float uArcHeight;
  uniform float uStartAngle;
  uniform sampler2D uDisplacementTexture;
  uniform float uDisplacementScale;
  uniform float uNoiseScale;
  uniform float uNoiseAmplitude;
  uniform float uTime;
  uniform bool uMirrorTexture;
  
  varying vec2 vUv;
  varying vec3 vPosition;
  varying vec4 vWorldPosition;
  varying vec3 vNormal;
  varying float vArcProgress;
  varying vec2 vMirrorUv;
  
  vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
  vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
  
  float snoise(vec3 v) { 
    const vec2 C = vec2(1.0/6.0, 1.0/3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i  = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;
    i = mod289(i);
    vec4 p = permute(permute(permute(
               i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
             + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
             + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
    float n_ = 0.142857142857;
    vec3  ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );
    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;
    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                  dot(p2,x2), dot(p3,x3) ) );
  }
  
  void main() {
    vUv = uv;
    vNormal = normal;
    float arcProgress = uv.x;
    vArcProgress = arcProgress;
    float radialOffset = (uv.y - 0.5) * uArcWidth;
    float phi = radians(uStartAngle) + arcProgress * radians(uArcAngle);
    vec3 arcPosition;
    arcPosition.x = cos(phi) * (uArcRadius + radialOffset);
    arcPosition.z = sin(phi) * (uArcRadius + radialOffset);
    arcPosition.y = (uv.y - 0.5) * uArcHeight;
    if (uMirrorTexture) {
        vMirrorUv = vec2(1.0 - uv.x, uv.y);
    } else {
        vMirrorUv = uv;
    }
    vec2 dispUv = uMirrorTexture ? vMirrorUv : uv;
    float displacement = texture2D(uDisplacementTexture, dispUv).r * uDisplacementScale;
    float cycleTime = abs(fract(uTime * 0.02) * 2.0 - 1.0); // Период ~50 сек
    float noise = snoise(vec3(arcPosition * uNoiseScale)) * uNoiseAmplitude;
    displacement += noise;
    vec3 displacedPosition = arcPosition + normalize(arcPosition) * displacement * 0.3;
    vPosition = displacedPosition;
    vWorldPosition = modelMatrix * vec4(displacedPosition, 1.0);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(displacedPosition, 1.0);
  }
`;

// ФРАГМЕНТНЫЙ ШЕЙДЕР
const fragmentShader = `
  uniform sampler2D uColorTexture;
  uniform sampler2D uAlphaTexture;
  uniform sampler2D uEmissionTexture;
  uniform float uOpacity;
  uniform float uTime;
  uniform float uFlowSpeed;
  uniform vec2 uFlowDirection;
  uniform float uColorModAmplitude;
  uniform float uArcAngle;
  uniform bool uMirrorTexture;
  
  varying vec2 vUv;
  varying vec3 vPosition;
  varying vec4 vWorldPosition;
  varying vec3 vNormal;
  varying float vArcProgress;
  varying vec2 vMirrorUv;
  
  vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
  vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
  
  float snoise(vec3 v) { 
    const vec2 C = vec2(1.0/6.0, 1.0/3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i  = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;
    i = mod289(i);
    vec4 p = permute(permute(permute(
               i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
             + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
             + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
    float n_ = 0.142857142857;
    vec3  ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );
    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;
    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                  dot(p2,x2), dot(p3,x3) ) );
  }
  
  void main() {
    vec2 baseUv = uMirrorTexture ? vMirrorUv : vUv;
    
    float flowCycle = abs(fract(uTime * uFlowSpeed * 0.02) * 2.0 - 1.0);
    vec2 arcFlowUv = vec2(
        uMirrorTexture ? (1.0 - vArcProgress) : vArcProgress,
        baseUv.y + uFlowDirection.y * flowCycle * 0.3
    ) + vec2(uFlowDirection.x * flowCycle * 0.3, 0.0);
    
    float time1 = flowCycle;
    float noiseLayer1 = snoise(vec3(baseUv * 1.5, time1)) * 0.001;
    vec2 chaoticOffset = vec2(noiseLayer1, 0.0);
    
    vec2 animatedUv = mod(arcFlowUv + chaoticOffset, 1.0);
    vec4 color = texture2D(uColorTexture, animatedUv);
    float alpha = texture2D(uAlphaTexture, animatedUv).r;
    vec3 emission = texture2D(uEmissionTexture, animatedUv).rgb;
    
    float colorTime1 = flowCycle;
    float colorNoise1 = (snoise(vec3(animatedUv * 4.0, colorTime1)) - 0.5) * 2.0 * uColorModAmplitude;
    vec3 finalColor = color.rgb + emission * (1.0 + colorNoise1);
    
    vec3 viewDir = normalize(vWorldPosition.xyz - cameraPosition);
    vec3 lightDir = normalize(vec3(0.2, 0.8, 0.3));
    float lighting = dot(normalize(vNormal), lightDir) * 0.6 + 0.4;
    finalColor *= lighting;
    
    float pulseTime = flowCycle;
    float glowPulse = sin(pulseTime) * 0.2 + 0.8;
    float cloudDensity = alpha * uOpacity * 1.8;
    cloudDensity = pow(cloudDensity, 0.75);
    finalColor += emission * cloudDensity * glowPulse * 0.8;
    
    gl_FragColor = vec4(finalColor, cloudDensity);
    if (cloudDensity < 0.03) {
      discard;
    }
    
    float seamMask = 1.0;
    float edgeFadeX = 1.0;
    if (baseUv.x < 0.2) edgeFadeX = smoothstep(0.0, 1.0, baseUv.x / 0.2);
    else if (baseUv.x > 0.8) edgeFadeX = smoothstep(0.0, 1.0, (1.0 - baseUv.x) / 0.2);
    float edgeFadeY = 1.0;
    if (baseUv.y < 0.2) edgeFadeY = smoothstep(0.0, 1.0, baseUv.y / 0.2);
    else if (baseUv.y > 0.8) edgeFadeY = smoothstep(0.0, 1.0, (1.0 - baseUv.y) / 0.2);
    seamMask = edgeFadeX * edgeFadeY * 0.8;
    cloudDensity *= seamMask;
  }
`;

// ФУНКЦИЯ СОЗДАНИЯ СТАТИЧНОЙ ДУГОВОЙ ТУМАННОСТИ
export const createNebula = (textures, config) => {
    const { colorTexture, alphaTexture, emissionTexture } = textures;
    const arcGeometry = new THREE.BufferGeometry();
    const arcAngleRad = THREE.MathUtils.degToRad(config.arcAngle);
    const startAngleRad = THREE.MathUtils.degToRad(config.startAngle);
    const segments = 64;
    const radialSegments = 16;
    const vertices = [];
    const uvs = [];
    const normals = [];
    const indices = [];
    for (let j = 0; j <= radialSegments; j++) {
        const radialProgress = j / radialSegments;
        const radialOffset = (radialProgress - 0.5) * config.arcWidth;
        for (let i = 0; i <= segments; i++) {
            const arcProgress = i / segments;
            const phi = startAngleRad + arcProgress * arcAngleRad;
            const radius = config.arcRadius + radialOffset;
            const x = Math.cos(phi) * radius;
            const z = Math.sin(phi) * radius;
            const y = (radialProgress - 0.5) * config.arcHeight;
            vertices.push(x, y, z);
            uvs.push(arcProgress, radialProgress);
            const normalX = Math.cos(phi);
            const normalZ = Math.sin(phi);
            normals.push(normalX, 0.0, normalZ);
        }
    }
    for (let j = 0; j < radialSegments; j++) {
        for (let i = 0; i < segments; i++) {
            const a = j * (segments + 1) + i;
            const b = j * (segments + 1) + i + 1;
            const c = (j + 1) * (segments + 1) + i;
            const d = (j + 1) * (segments + 1) + i + 1;
            indices.push(a, b, d);
            indices.push(a, d, c);
        }
    }
    arcGeometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    arcGeometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    arcGeometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
    arcGeometry.setIndex(indices);
    arcGeometry.computeVertexNormals();
    const material = new THREE.ShaderMaterial({
        uniforms: {
            uColorTexture: { value: colorTexture },
            uAlphaTexture: { value: alphaTexture },
            uEmissionTexture: { value: emissionTexture },
            uDisplacementTexture: { value: alphaTexture },
            uDisplacementScale: { value: config.displacementScale },
            uNoiseScale: { value: config.noiseScale },
            uNoiseAmplitude: { value: config.noiseAmplitude },
            uFlowSpeed: { value: config.flowSpeed },
            uFlowDirection: { value: new THREE.Vector2(config.flowDirection.x, config.flowDirection.y) },
            uColorModAmplitude: { value: config.colorModAmplitude },
            uTime: { value: 0 },
            uOpacity: { value: config.opacity },
            uArcAngle: { value: config.arcAngle },
            uArcRadius: { value: config.arcRadius },
            uArcWidth: { value: config.arcWidth },
            uArcHeight: { value: config.arcHeight },
            uStartAngle: { value: config.startAngle },
            uMirrorTexture: { value: config.mirrorTexture || false }
        },
        vertexShader,
        fragmentShader,
        transparent: true,
        side: THREE.DoubleSide,
        depthWrite: false,
        depthTest: true,
        blending: THREE.NormalBlending
    });
    const nebula = new THREE.Mesh(arcGeometry, material);
    nebula.position.set(0, config.yOffset || -0.5, 0);
    nebula.scale.set(1.2, 1.0, 1.2);
    nebula.rotation.x = THREE.MathUtils.degToRad(1 + Math.random() * 3);
    nebula.rotation.z = THREE.MathUtils.degToRad(-1 + Math.random() * 3);
    [colorTexture, alphaTexture, emissionTexture].forEach(texture => {
        texture.wrapS = THREE.RepeatWrapping;
        texture.wrapT = THREE.RepeatWrapping;
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
    });
    nebula.userData = {
        config,
        update: (time) => {
            material.uniforms.uTime.value = time;
        }
    };
    const customDepthMaterial = new THREE.MeshDepthMaterial({
        depthPacking: THREE.RGBADepthPacking,
        alphaMap: alphaTexture,
        opacity: config.opacity
    });
    nebula.customDepthMaterial = customDepthMaterial;
    nebula.castShadow = true;
    nebula.receiveShadow = true;
    const innerRadius = config.arcRadius - config.arcWidth / 2;
    const outerRadius = config.arcRadius + config.arcWidth / 2;
    const mirrorMode = config.mirrorTexture ? 'ЗЕРКАЛЬНАЯ' : 'ОРДИНАРНАЯ';
    console.log(`🌈 ${mirrorMode} дуговая туманность "${config.name}": дуга ${config.arcAngle}° от ${config.startAngle}°, покрытие ${innerRadius.toFixed(1)}-${outerRadius.toFixed(1)}`);
    return nebula;
};

// Загрузка текстур
export const loadNebulaTextures = async (textureLoader, config) => {
    try {
        const [colorTexture, alphaTexture, emissionTexture] = await Promise.all([
            new Promise((resolve) => textureLoader.load(config.colorPath, resolve)),
            new Promise((resolve) => textureLoader.load(config.alphaPath, resolve)),
            new Promise((resolve) => textureLoader.load(config.emissionPath, resolve))
        ]);
        [colorTexture, alphaTexture, emissionTexture].forEach(texture => {
            texture.wrapS = THREE.RepeatWrapping;
            texture.wrapT = THREE.RepeatWrapping;
            texture.minFilter = THREE.LinearFilter;
            texture.magFilter = THREE.LinearFilter;
        });
        return { colorTexture, alphaTexture, emissionTexture };
    } catch (error) {
        console.error(`Ошибка загрузки текстур для ${config.name}:`, error);
        return null;
    }
};

// Загрузка и создание всех статичных дуговых туманностей
export const loadAllNebulas = async (textureLoader) => {
    const nebulas = [];
    for (const config of nebulaConfigs) {
        const textures = await loadNebulaTextures(textureLoader, config);
        if (textures) {
            const nebula = createNebula(textures, config);
            nebulas.push(nebula);
        }
    }
    console.log(`✨ Загружено ${nebulas.length} статичных дуговых туманностей!`);
    return nebulas;
};

// Обновление всех туманностей
export const updateNebulas = (nebulas, time) => {
    nebulas.forEach(nebula => {
        if (nebula.userData && nebula.userData.update) {
            nebula.userData.update(time);
        }
    });
};