import React from 'react';
import { ArrowLeftIcon, SparklesIcon, ChartBarIcon } from '@heroicons/react/24/solid';
import TopUserProfile from './TopUserProfile';

const Header = ({
                    onBack,
                    showChatLabel,
                    showBackButton,
                    showStats,
                    onToggleStats,
                    isMobile
                }) => {
    return (
        <div className="sticky top-0 w-full z-50 px-4 py-3 bg-gradient-to-b from-black/50 via-black/30 to-transparent">
            <div className="relative flex items-center justify-between w-full">
                {/* Левая часть - кнопки */}
                <div className="flex items-center gap-2 z-10">
                    {/* Кнопка статистики */}
                    <button
                        onClick={onToggleStats}
                        data-stats-button="true"
                        className={`w-10 h-10 btn-universal p-2 !rounded-full transition-all duration-300 ${
                            showStats ? 'gr2' : 'gr1'
                        } hover:scale-105 active:scale-95`}
                        title={showStats ? 'Скрыть статистику' : 'Показать статистику'}
                    >
                        <ChartBarIcon className={`w-5 h-5 transition-all duration-300 ${
                            showStats ? 'text-yellow-400' : 'text-white'
                        }`} />
                    </button>

                    {/* Кнопка назад */}
                    {showBackButton && (
                        <button
                            onClick={onBack}
                            className="w-10 h-10 btn-universal gr1 p-2 !rounded-full transition-all duration-300 hover:scale-110 active:scale-95"
                        >
                            <ArrowLeftIcon className="w-5 h-5 text-white ml-0.5 transition-transform duration-300 hover:-translate-x-0.5" />
                        </button>
                    )}
                </div>

                {/* Центральная часть - заголовок (абсолютно по центру) */}
                <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
                    <h1 className="flex items-center gap-1 text-xl font-bold text-white whitespace-nowrap">
                        Serenity <SparklesIcon className="w-6 h-6" />
                        {showChatLabel && 'чат'}
                    </h1>
                </div>

                {/* Правая часть - профиль */}
                <div className="z-10">
                    <TopUserProfile />
                </div>
            </div>
        </div>
    );
};

export default Header;
