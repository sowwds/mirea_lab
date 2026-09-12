import React, { useState, useEffect, useMemo } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader';
import { planetConfigs } from './planetLoader';
import {
  emotionsDataByTime,
  emotionsList,
  emotionToPlanetIndex,
  getGradient,
  getEmotionColor,
} from '../services/emotionData';

const imageCache = {};

// ── Animated progress bar ─────────────────────────────────────────────────────
const AnimatedBar = ({ value, gradient, periodKey }) => {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    setWidth(0);
    const t = setTimeout(() => setWidth(value), 50);
    return () => clearTimeout(t);
  }, [value, periodKey]);
  return (
    <div className="bg-white/10 rounded-full h-2 overflow-hidden">
      <div
        className="h-2 rounded-full transition-all duration-700 ease-out"
        style={{ width: `${width}%`, background: gradient }}
      />
    </div>
  );
};

// ── Planet icon — renders GLB to canvas, same logic as EmotionalStats ─────────
const PlanetIcon2D = ({ planet, renderer }) => {
  const loadPath =
    planet.emotion === 'sadness' && planet.iconPath
      ? planet.iconPath
      : planet.modelPath;

  const [imageUrl, setImageUrl] = useState(imageCache[loadPath] || null);
  const [loading, setLoading] = useState(!imageCache[loadPath]);

  useEffect(() => {
    if (imageCache[loadPath]) {
      setImageUrl(imageCache[loadPath]);
      setLoading(false);
      return;
    }

    const loader = new GLTFLoader();
    const draco = new DRACOLoader();
    draco.setDecoderPath('/draco/');
    loader.setDRACOLoader(draco);

    loader.load(
      loadPath,
      (gltf) => {
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 1000);
        camera.position.set(0, 0, 1.5);

        let targetModel = gltf.scene;
        if (planet.emotion === 'sadness') {
          let main = null, maxVol = 0;
          gltf.scene.traverse((c) => {
            if (c.isMesh || c.isGroup) {
              const sz = new THREE.Box3().setFromObject(c).getSize(new THREE.Vector3());
              const vol = sz.x * sz.y * sz.z;
              if (vol > maxVol) { maxVol = vol; main = c; }
            }
          });
          if (main) targetModel = main;
        }

        const model = targetModel.clone();
        model.position.set(0, 0, 0);
        model.rotation.set(0, 0, 0);
        model.scale.set(1, 1, 1);

        const box = new THREE.Box3().setFromObject(model);
        const sz = box.getSize(new THREE.Vector3());
        const center = new THREE.Vector3();
        box.getCenter(center);
        const s = (1.0 / Math.max(sz.x, sz.y, sz.z)) * 0.8;
        model.position.set(-center.x * s, -center.y * s, -center.z * s);
        model.scale.set(s, s, s);
        scene.add(model);

        const l1 = new THREE.DirectionalLight(0xffffff, 1);
        l1.position.set(1, 1, 1);
        scene.add(l1);
        const l2 = new THREE.DirectionalLight(0x72bbec, 1);
        l2.position.set(-2, -1, -1);
        scene.add(l2);

        renderer.setSize(64, 64);
        renderer.setClearColor(0x000000, 0);
        renderer.render(scene, camera);

        const url = renderer.domElement.toDataURL('image/png');
        imageCache[loadPath] = url;
        setImageUrl(url);
        setLoading(false);

        scene.remove(model, l1, l2);
      },
      undefined,
      () => {
        // Fallback — gradient circle
        const canvas = document.createElement('canvas');
        canvas.width = 64;
        canvas.height = 64;
        const ctx = canvas.getContext('2d');
        const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
        grad.addColorStop(0, getEmotionColor(planet.emotion, 1));
        grad.addColorStop(1, getEmotionColor(planet.emotion, 0.3));
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, 64, 64);
        const url = canvas.toDataURL('image/png');
        imageCache[loadPath] = url;
        setImageUrl(url);
        setLoading(false);
      },
    );

    return () => draco.dispose();
  }, [loadPath, planet.emotion, renderer]);

  if (loading) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-gray-800 rounded-full">
        <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <img
      src={imageUrl}
      alt={planet.name}
      className="w-full h-full rounded-full object-contain"
      style={{ backgroundColor: 'transparent' }}
    />
  );
};

// ── Main component ─────────────────────────────────────────────────────────────
const PERIODS = [
  { key: 'day',   label: 'День' },
  { key: 'week',  label: 'Неделя' },
  { key: 'month', label: 'Месяц' },
  { key: 'all',   label: 'Все время' },
];

const LandingStats = () => {
  const [period, setPeriod] = useState('week');

  const renderer = useMemo(() => {
    const r = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'low-power',
    });
    r.setSize(64, 64);
    r.setPixelRatio(1);
    return r;
  }, []);

  useEffect(() => {
    return () => { setTimeout(() => renderer.dispose(), 1000); };
  }, [renderer]);

  return (
    <div className="bg-black/20 backdrop-blur-xl border border-white/10 rounded-2xl p-5 w-full max-w-sm shadow-2xl shadow-black/40">
      <h3 className="text-white font-bold text-sm uppercase tracking-widest text-center mb-3">
        Эмоциональный профиль
      </h3>
      <div className="h-px bg-white/20 mb-3" />

      {/* Period tabs */}
      <div className="flex gap-1 justify-center mb-4">
        {PERIODS.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setPeriod(key)}
            className={`px-2 py-1.5 rounded-md text-xs transition-all duration-300 ${
              period === key
                ? 'bg-white/20 text-white font-semibold shadow-[0_0_8px_rgba(255,255,255,0.2)]'
                : 'text-white/50 hover:text-white'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Emotion rows */}
      <div className="space-y-3">
        {emotionsList.map((key) => {
          const emotion = emotionsDataByTime[key];
          const { value, trend } = emotion[period];
          const planetIndex = emotionToPlanetIndex[key];
          const planet = planetConfigs[planetIndex] || {
            name: key,
            modelPath: '/models/placeholder.glb',
            emotion: key,
          };

          return (
            <div key={key} className="flex items-center gap-3 h-10">
              <div className="w-10 h-10 min-w-[2.5rem] relative">
                <PlanetIcon2D planet={planet} renderer={renderer} />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-white/80 text-xs truncate">{emotion.name}</span>
                  <div className="flex items-center gap-1.5 flex-shrink-0">
                    <span className="text-white text-xs font-medium">{value}%</span>
                    <span className={`text-xs ${
                      trend === 'up' ? 'text-green-400' : trend === 'down' ? 'text-red-400' : 'text-blue-400'
                    }`}>
                      {trend === 'up' ? '↑' : trend === 'down' ? '↓' : '→'}
                    </span>
                  </div>
                </div>
                <AnimatedBar value={value} gradient={getGradient(key)} periodKey={period} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default LandingStats;
