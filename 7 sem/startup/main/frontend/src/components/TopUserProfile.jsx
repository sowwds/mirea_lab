import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { UserIcon, ArrowRightStartOnRectangleIcon } from '@heroicons/react/24/solid';
import { logout, getProfile } from '../services/api';
import { useAuth } from '../context/AuthContext';

const TopUserProfile = () => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  const [profile, setProfile] = useState({ fullName: 'Профиль', email: 'Не указано' });
  const dropdownRef = useRef(null);
  const navigate = useNavigate();
  const { handleLogout } = useAuth();

  // Fetch profile data on mount
  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const profileData = await getProfile();
        setProfile(profileData);
      } catch (e) {
        console.error('TopUserProfile: Fetch profile error', e);
      }
    };
    fetchProfile();
  }, []);

  // Handle window resize to update isMobile
  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Handle clicks outside to collapse dropdown on mobile
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (isMobile && isExpanded && dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        console.log('Click outside detected, closing dropdown');
        setIsExpanded(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isExpanded, isMobile]);

  // Toggle expanded state on click for mobile
  const handleClick = () => {
    if (isMobile) {
      console.log('Toggling expanded state');
      setIsExpanded(!isExpanded);
    }
  };

  // Handle hover for desktop
  const handleMouseEnter = () => {
    if (!isMobile) {
      setIsExpanded(true);
    }
  };

  // Handle hover for desktop
  const handleMouseLeave = () => {
    if (!isMobile) {
      setIsExpanded(false);
    }
  };

  // Handle logout
  const handleLogoutClick = async () => {
    try {
      console.log('TopUserProfile: Logging out');
      await logout();
      localStorage.removeItem('access_token');
      console.log('TopUserProfile: Token cleared, navigating to /auth');
      handleLogout();
      navigate('/auth', { replace: true });
    } catch (e) {
      console.error('TopUserProfile: Logout error', e);
      localStorage.removeItem('access_token');
      console.log('TopUserProfile: Token cleared (on error), navigating to /auth');
      handleLogout();
      navigate('/auth', { replace: true });
    }
  };

  return (
    <div className="relative h-10 lg:w-fit">
      <div
        ref={dropdownRef}
        className={`absolute top-0 right-0 btn-universal gr5 cursor-pointer !rounded-3xl z-10 backdrop-blur-md transition-all duration-300 ease-in-out transform-origin-top-right ${
          isExpanded
            ? 'active bg-gradient-to-r from-[#fbc2eb] to-[#a6c1ee] max-h-[340px] max-w-[256px] shadow-[0_0_15px_rgba(255,255,255,0.3),0_0_30px_rgba(255,255,255,0.2)] scale-105'
            : 'bg-[rgba(49,50,68,0.5)] max-h-10 max-w-[40px] lg:max-w-fit'
        }`}
        onClick={handleClick}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        {!isExpanded ? (
          <div className="flex items-center gap-2 p-2 h-10">
            <UserIcon className="w-6 h-6 text-white" />
            <span className="hidden lg:block text-white truncate max-w-[100px] truncate">
              {profile.fullName}
            </span>
          </div>
        ) : (
          <div className="p-4 space-y-3">
            <div className="space-y-2">
              <p className="text-sm text-gray-800">Аккаунт</p>
              <div className="flex items-center gap-3">
                <UserIcon className="w-10 h-10 text-white p-2 rounded-full bg-gray-600/50" />
                <div>
                  <p className="font-medium text-md text-gray-700 truncate">
                    {profile.fullName}
                  </p>
                  <p className="text-xs text-gray-700 truncate max-w-[100px] truncate">
                    {profile.email}
                  </p>
                </div>
              </div>
            </div>
            <div className="border-t border-subtext0/50 my-2"></div>
            <button
              onClick={handleLogoutClick}
              className="w-full flex items-center justify-center gap-2 px-3 py-2 text-red-500 hover:scale-105 active:scale-95 transition-transform duration-200 rounded-lg bg-surface0/30 hover:bg-surface0/40"
            >
              <ArrowRightStartOnRectangleIcon className="h-5 w-5" />
              Выйти
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default TopUserProfile;
