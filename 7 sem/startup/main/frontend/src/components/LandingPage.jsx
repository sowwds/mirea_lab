import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import gsap from 'gsap';
import { SparklesIcon, ChevronDownIcon } from '@heroicons/react/24/solid';
import LandingChat from './LandingChat';
import LandingStats from './LandingStats';

// Intersection Observer hook for scroll-triggered animations
function useScrollAnimations() {
  useEffect(() => {
    const els = document.querySelectorAll('.scroll-animate');
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach(e => {
          if (e.isIntersecting) {
            e.target.classList.add('visible');
          }
        });
      },
      { threshold: 0.12 }
    );
    els.forEach(el => observer.observe(el));
    return () => observer.disconnect();
  }, []);
}

const FAQ_ITEMS = [
  {
    q: 'Может ли ИИ заменить настоящего психолога?',
    a: 'Нет. Serenity — не замена профессиональной психологической помощи. Это инструмент самопознания и эмоциональной поддержки, доступный в любое время. При серьёзных проблемах мы рекомендуем обратиться к специалисту.',
  },
  {
    q: 'Насколько это анонимно и безопасно?',
    a: 'Полностью. Мы не требуем реального имени, телефона или email. Все данные шифруются и никогда не передаются третьим лицам.',
  },
  {
    q: 'На каких научных методах основан сервис?',
    a: 'В основе лежат принципы когнитивно-поведенческой терапии (КПТ) и позитивной психологии. Наша модель обучена под руководством практикующих психологов.',
  },
  {
    q: 'Как я могу отменить подписку?',
    a: 'В любой момент из личного кабинета, без объяснения причин. Никаких скрытых условий.',
  },
  {
    q: 'Можно ли пользоваться с мобильного телефона?',
    a: 'Да, сервис полностью адаптирован для мобильных устройств.',
  },
];

const FaqItem = ({ question, answer }) => {
  const [open, setOpen] = useState(false);
  return (
    <div className="border-b border-white/10">
      <button
        className="w-full flex justify-between items-center py-5 text-left gap-4 cursor-pointer"
        onClick={() => setOpen(o => !o)}
      >
        <span className="text-white/85 font-medium text-base">{question}</span>
        <ChevronDownIcon
          className={`w-5 h-5 text-white/40 flex-shrink-0 transition-transform duration-300 ${open ? 'rotate-180' : ''}`}
        />
      </button>
      <div
        className="overflow-hidden transition-all duration-300 ease-in-out"
        style={{ maxHeight: open ? '200px' : '0px' }}
      >
        <p className="text-white/50 text-sm leading-relaxed pb-5">{answer}</p>
      </div>
    </div>
  );
};

const FEATURES = [
  {
    icon: '🕐',
    title: 'Доступность 24/7',
    text: 'Сервис доступен в любое время дня и ночи, когда вам нужна поддержка или совет, в отличие от психолога.',
  },
  {
    icon: '🔒',
    title: 'Конфиденциальный диалог',
    text: 'Наш ИИ обучен вести диалог, помогая вам разобраться в мыслях и чувствах. Все данные надёжно шифруются.',
  },
  {
    icon: '💙',
    title: 'Полная анонимность',
    text: 'Мы гарантируем полную анонимность. Вам не нужно делиться личными данными, чтобы получить помощь.',
  },
];

const FREE_FEATURES    = ['До 10 сессий в месяц', 'Базовый дневник эмоций', 'Стандартная аналитика'];
const PREMIUM_FEATURES = ['Безлимитные сессии', 'Продвинутая аналитика', 'Глубокие инсайты', 'Экспорт данных', 'Приоритетная поддержка'];

export default function LandingPage() {
  const navigate = useNavigate();
  const heroContentRef = useRef(null);
  const [scrolled, setScrolled] = useState(false);

  useScrollAnimations();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 60);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  useEffect(() => {
    if (!heroContentRef.current) return;
    const children = Array.from(heroContentRef.current.children);
    gsap.fromTo(
      children,
      { y: 32, opacity: 0 },
      { y: 0, opacity: 1, duration: 0.9, stagger: 0.14, ease: 'power2.out', delay: 0.15 }
    );
  }, []);

  const scrollTo = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen bg-[#0A0E21] text-white">

      {/* ── NAVBAR ── */}
      <nav
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
          scrolled
            ? 'bg-black/60 backdrop-blur-xl border-b border-white/10 shadow-lg shadow-black/40'
            : 'bg-transparent'
        }`}
      >
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between gap-6">
          <button
            onClick={() => scrollTo('hero')}
            className="text-xl font-bold flex items-center gap-1.5 cursor-pointer"
          >
            Serenity <SparklesIcon className="w-5 h-5 opacity-80" />
          </button>

          <div className="hidden md:flex gap-8 flex-1 justify-center">
            {[
              ['features', 'О сервисе'],
              ['chat',     'Возможности'],
              ['pricing',  'Тарифы'],
              ['faq',      'FAQ'],
            ].map(([id, label]) => (
              <button
                key={id}
                onClick={() => scrollTo(id)}
                className="text-white/60 hover:text-white text-sm transition-colors duration-200 cursor-pointer"
              >
                {label}
              </button>
            ))}
          </div>

          <button
            onClick={() => navigate('/auth')}
            className="btn-universal px-5 py-2 text-sm font-semibold flex-shrink-0"
          >
            Войти →
          </button>
        </div>
      </nav>

      {/* ── 1. HERO ── */}
      <section id="hero" className="relative min-h-[92vh] flex items-center overflow-hidden">
        {/* Background image + overlays */}
        <img
          src="/exampleimage2.jpg"
          alt=""
          className="absolute inset-0 w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-r from-[#0A0E21]/95 via-[#0A0E21]/65 to-[#0A0E21]/25" />
        {/* Strong bottom bleed — fades into next section colour */}
        <div className="absolute inset-0 bg-gradient-to-t from-[#050716] via-[#050716]/60 to-transparent" />

        {/* Content — constrained by same max-w as navbar so it doesn't hug the edge */}
        <div className="relative z-10 w-full max-w-7xl mx-auto px-6 md:px-16">
        <div ref={heroContentRef} className="max-w-[600px]">
          <span className="inline-block text-xs font-semibold tracking-[3px] uppercase text-blue-300/70 mb-4">
            Serenity AI
          </span>
          <h1 className="text-5xl md:text-[3.6rem] font-extrabold leading-[1.08] tracking-tight mb-5">
            Понимание себя начинается здесь.
          </h1>
          <p className="text-lg text-white/60 mb-8 leading-relaxed">
            Ваш личный ИИ-психолог.<br />Анонимно. Доступно. 24/7.
          </p>
          <button
            onClick={() => navigate('/auth')}
            className="btn-universal gr6 px-8 py-4 text-lg font-semibold"
          >
            Начать бесплатно →
          </button>
        </div>
        </div>
      </section>

      {/* ── 2. FEATURES ── */}
      <section id="features" className="py-24 px-6 bg-[#050716]">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-14 scroll-animate">
            <span className="text-xs font-semibold tracking-[3px] uppercase text-blue-300/70 mb-3 block">
              Почему Serenity
            </span>
            <h2 className="text-4xl font-bold">Поддержка, основанная на технологиях</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {FEATURES.map((f, i) => (
              <div
                key={i}
                className="scroll-animate bg-white/[0.04] border border-white/10 rounded-2xl p-7 backdrop-blur-sm text-center"
                style={{ transitionDelay: `${i * 0.1}s` }}
              >
                <div className="text-4xl mb-4">{f.icon}</div>
                <h3 className="text-lg font-semibold mb-2">{f.title}</h3>
                <p className="text-white/50 text-sm leading-relaxed">{f.text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── 3. CHAT DEMO ── */}
      <section id="chat" className="relative py-24 px-6 overflow-hidden bg-[#050716]">
        {/* Atmospheric right background */}
        <div className="absolute right-0 top-0 bottom-0 w-1/2 pointer-events-none select-none overflow-hidden">
          <img
            src="/example9-16.png"
            alt=""
            className="h-full w-full object-cover opacity-25"
            style={{
              WebkitMaskImage: 'linear-gradient(to right, transparent 0%, rgba(0,0,0,0.8) 45%)',
              maskImage:        'linear-gradient(to right, transparent 0%, rgba(0,0,0,0.8) 45%)',
            }}
          />
        </div>
        {/* Top + bottom section bleed — blends with adjacent sections */}
        <div className="absolute top-0 left-0 right-0 h-36 bg-gradient-to-b from-[#050716] to-transparent pointer-events-none z-[5]" />
        <div className="absolute bottom-0 left-0 right-0 h-36 bg-gradient-to-t from-[#050716] to-transparent pointer-events-none z-[5]" />

        <div className="relative z-10 max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          <div className="scroll-animate">
            <span className="text-xs font-semibold tracking-[3px] uppercase text-blue-300/70 mb-3 block">
              Возможность 1
            </span>
            <h2 className="text-4xl font-bold mb-4 leading-tight">Ваш личный собеседник</h2>
            <p className="text-white/55 text-base leading-relaxed mb-8">
              Обсуждайте то, что вас волнует, без осуждения и страха. Наш ИИ поможет найти
              корень проблемы и просто выслушает — как друг, который <em>всегда</em> рядом.
            </p>
            <button
              onClick={() => navigate('/auth')}
              className="btn-universal gr5 px-6 py-3 font-semibold"
            >
              Начать диалог →
            </button>
          </div>

          <div className="scroll-animate" style={{ transitionDelay: '0.15s' }}>
            <LandingChat />
          </div>
        </div>
      </section>

      {/* ── 4. STATS DEMO ── */}
      <section id="stats" className="relative py-24 px-6 overflow-hidden bg-[#050716]">
        {/* Atmospheric left background */}
        <div className="absolute left-0 top-0 bottom-0 w-1/2 pointer-events-none select-none overflow-hidden">
          <img
            src="/exampleimage3.jpg"
            alt=""
            className="h-full w-full object-cover opacity-25"
            style={{
              WebkitMaskImage: 'linear-gradient(to left, transparent 0%, rgba(0,0,0,0.8) 45%)',
              maskImage:        'linear-gradient(to left, transparent 0%, rgba(0,0,0,0.8) 45%)',
            }}
          />
        </div>
        {/* Top + bottom bleed */}
        <div className="absolute top-0 left-0 right-0 h-36 bg-gradient-to-b from-[#050716] to-transparent pointer-events-none z-[5]" />
        <div className="absolute bottom-0 left-0 right-0 h-36 bg-gradient-to-t from-[#050716] to-transparent pointer-events-none z-[5]" />

        <div className="relative z-10 max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          <div className="scroll-animate flex justify-center lg:justify-start">
            <LandingStats />
          </div>

          <div className="scroll-animate" style={{ transitionDelay: '0.15s' }}>
            <span className="text-xs font-semibold tracking-[3px] uppercase text-blue-300/70 mb-3 block">
              Возможность 2
            </span>
            <h2 className="text-4xl font-bold mb-4 leading-tight">
              Наглядные отчёты о вашем прогрессе
            </h2>
            <p className="text-white/55 text-base leading-relaxed mb-8">
              Отслеживайте динамику своего эмоционального состояния через интерактивные
              графики и получайте персональные инсайты каждую неделю.
            </p>
            <button
              onClick={() => navigate('/auth')}
              className="btn-universal gr3 px-6 py-3 font-semibold"
            >
              Начать отслеживать →
            </button>
          </div>
        </div>
      </section>

      {/* ── 5. PRICING ── */}
      <section id="pricing" className="py-24 px-6 bg-[#050716]">
        <div className="max-w-4xl mx-auto text-center">
          <div className="scroll-animate mb-14">
            <span className="text-xs font-semibold tracking-[3px] uppercase text-blue-300/70 mb-3 block">
              Тарифы
            </span>
            <h2 className="text-4xl font-bold">Выберите свой план</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-2xl mx-auto">
            {/* Free */}
            <div className="scroll-animate bg-white/[0.04] border border-white/10 rounded-2xl p-8 text-left flex flex-col">
              <p className="text-white/50 text-sm mb-4">Базовый</p>
              <div className="text-4xl font-extrabold mb-1">Бесплатно</div>
              <div className="text-white/40 text-xs mb-6">навсегда</div>
              <ul className="space-y-2 text-sm text-white/60 flex-1">
                {FREE_FEATURES.map(f => (
                  <li key={f} className="flex items-center gap-2">
                    <span className="text-blue-300">✓</span>{f}
                  </li>
                ))}
              </ul>
              <button
                onClick={() => navigate('/auth')}
                className="btn-universal w-full py-3 mt-6 font-semibold"
              >
                Начать бесплатно
              </button>
            </div>

            {/* Premium */}
            <div
              className="scroll-animate bg-blue-300/[0.05] border border-blue-300/30 rounded-2xl p-8 text-left relative flex flex-col"
              style={{ transitionDelay: '0.1s' }}
            >
              <div className="absolute -top-3 right-6 bg-gradient-to-r from-blue-500 to-purple-600 text-white text-xs font-bold px-3 py-1 rounded-full shadow-lg">
                Популярный
              </div>
              <p className="text-white/50 text-sm mb-4">Премиум</p>
              <div className="text-4xl font-extrabold mb-1">499 ₽</div>
              <div className="text-white/40 text-xs mb-6">в месяц</div>
              <ul className="space-y-2 text-sm text-white/60 flex-1">
                {PREMIUM_FEATURES.map(f => (
                  <li key={f} className="flex items-center gap-2">
                    <span className="text-blue-300">✓</span>{f}
                  </li>
                ))}
              </ul>
              <button
                onClick={() => navigate('/auth')}
                className="btn-universal gr6 w-full py-3 mt-6 font-semibold"
              >
                Начать с Премиум
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* ── 6. FAQ ── */}
      <section id="faq" className="py-24 px-6 bg-[#050716]">
        <div className="max-w-3xl mx-auto">
          <div className="scroll-animate mb-12">
            <span className="text-xs font-semibold tracking-[3px] uppercase text-blue-300/70 mb-3 block">
              FAQ
            </span>
            <h2 className="text-4xl font-bold">Ответы на частые вопросы</h2>
          </div>
          <div className="scroll-animate">
            {FAQ_ITEMS.map(({ q, a }) => (
              <FaqItem key={q} question={q} answer={a} />
            ))}
          </div>
        </div>
      </section>

      {/* ── 7. FINAL CTA ── */}
      <section className="py-28 px-6 text-center bg-[#050716] relative overflow-hidden">
        {/* Glow */}
        <div
          className="absolute inset-0 flex items-center justify-center pointer-events-none"
          aria-hidden
        >
          <div className="w-[600px] h-[600px] rounded-full bg-purple-700/15 blur-[100px]" />
        </div>

        <div className="relative z-10 scroll-animate">
          <h2 className="text-5xl font-extrabold mb-4">Попробуйте прямо сейчас!</h2>
          <p className="text-white/50 text-lg mb-10">
            Просто начните набирать сообщение — первые 10 сессий бесплатно.
          </p>
          <button
            onClick={() => navigate('/auth')}
            className="btn-universal gr6 px-10 py-5 text-xl font-bold"
          >
            Зарегистрироваться бесплатно →
          </button>
        </div>
      </section>

      {/* ── FOOTER ── */}
      <footer className="py-8 px-6 bg-[#050716] border-t border-white/10 text-center text-white/30 text-sm">
        © 2025 Serenity AI · Все права защищены
      </footer>
    </div>
  );
}
