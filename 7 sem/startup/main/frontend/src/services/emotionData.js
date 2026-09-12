const emotionsDataByTime = {
  joy: { day: { value: 65, trend: 'up' }, week: { value: 72, trend: 'stable' }, month: { value: 68, trend: 'down' }, all: { value: 75, trend: 'up' }, name: 'Радость', emotion: 'joy' },
  sadness: { day: { value: 40, trend: 'stable' }, week: { value: 45, trend: 'down' }, month: { value: 50, trend: 'stable' }, all: { value: 40, trend: 'down' }, name: 'Грусть', emotion: 'sadness' },
  anger: { day: { value: 15, trend: 'down' }, week: { value: 20, trend: 'stable' }, month: { value: 10, trend: 'down' }, all: { value: 15, trend: 'down' }, name: 'Гнев', emotion: 'anger' },
  fear: { day: { value: 25, trend: 'stable' }, week: { value: 30, trend: 'up' }, month: { value: 20, trend: 'down' }, all: { value: 25, trend: 'stable' }, name: 'Страх', emotion: 'fear' },
  surprise: { day: { value: 65, trend: 'up' }, week: { value: 70, trend: 'stable' }, month: { value: 60, trend: 'up' }, all: { value: 65, trend: 'up' }, name: 'Удивление', emotion: 'surprise' },
  trust: { day: { value: 85, trend: 'up' }, week: { value: 80, trend: 'stable' }, month: { value: 90, trend: 'up' }, all: { value: 85, trend: 'up' }, name: 'Доверие', emotion: 'trust' },
  anticipation: { day: { value: 50, trend: 'stable' }, week: { value: 55, trend: 'up' }, month: { value: 45, trend: 'down' }, all: { value: 50, trend: 'stable' }, name: 'Ожидание', emotion: 'anticipation' },
  disgust: { day: { value: 10, trend: 'down' }, week: { value: 15, trend: 'stable' }, month: { value: 5, trend: 'down' }, all: { value: 10, trend: 'down' }, name: 'Отвращение', emotion: 'disgust' }
};

const emotionsList = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'trust', 'anticipation', 'disgust'];

const emotionToPlanetIndex = {
  joy: 1,
  sadness: 5,
  anger: 7,
  fear: 3,
  surprise: 4,
  trust: 2,
  anticipation: 8,
  disgust: 6
};

const topicsByEmotion = {
  joy: [
    { date: '15.12.2023', summary: 'Вы выразили радость по поводу завершения проекта', impact: '+15%', emotion: 'радость' },
    { date: '10.12.2023', summary: 'Радость от встречи с друзьями', impact: '+12%', emotion: 'радость' },
    { date: '05.12.2023', summary: 'Празднование дня рождения', impact: '+18%', emotion: 'радость' },
    { date: '01.12.2023', summary: 'Успех в работе', impact: '+10%', emotion: 'радость' },
    { date: '25.11.2023', summary: 'Приятный сюрприз', impact: '+14%', emotion: 'радость' },
    { date: '20.11.2023', summary: 'Победа в соревновании', impact: '+16%', emotion: 'радость' },
    { date: '15.11.2023', summary: 'Хорошие новости от семьи', impact: '+11%', emotion: 'радость' },
    { date: '10.11.2023', summary: 'Отличный выходной', impact: '+13%', emotion: 'радость' },
    { date: '05.11.2023', summary: 'Радость от хобби', impact: '+9%', emotion: 'радость' },
    { date: '01.11.2023', summary: 'Позитивный отзыв', impact: '+17%', emotion: 'радость' }
  ],
  trust: [
    { date: '15.12.2023', summary: 'Обсуждали важность доверия в отношениях', impact: '+10%', emotion: 'доверие' },
    { date: '10.12.2023', summary: 'Доверие к коллегам', impact: '+8%', emotion: 'доверие' },
    { date: '05.12.2023', summary: 'Построение доверия в команде', impact: '+12%', emotion: 'доверие' },
    { date: '01.12.2023', summary: 'Доверительный разговор', impact: '+9%', emotion: 'доверие' },
    { date: '25.11.2023', summary: 'Доверие к партнеру', impact: '+11%', emotion: 'доверие' },
    { date: '20.11.2023', summary: 'Укрепление доверия', impact: '+13%', emotion: 'доверие' },
    { date: '15.11.2023', summary: 'Доверие в дружбе', impact: '+7%', emotion: 'доверие' },
    { date: '10.11.2023', summary: 'Доверительные отношения', impact: '+14%', emotion: 'доверие' },
    { date: '05.11.2023', summary: 'Построение доверия', impact: '+10%', emotion: 'доверие' },
    { date: '01.11.2023', summary: 'Доверие к себе', impact: '+15%', emotion: 'доверие' }
  ],
  fear: [
    { date: '15.12.2023', summary: 'Проявили беспокойство о будущем', impact: '-8%', emotion: 'страх' },
    { date: '10.12.2023', summary: 'Страх перед неизвестным', impact: '-10%', emotion: 'страх' },
    { date: '05.12.2023', summary: 'Беспокойство о здоровье', impact: '-7%', emotion: 'страх' },
    { date: '01.12.2023', summary: 'Страх неудачи', impact: '-9%', emotion: 'страх' },
    { date: '25.11.2023', summary: 'Тревога по поводу работы', impact: '-11%', emotion: 'страх' },
    { date: '20.11.2023', summary: 'Страх потери', impact: '-6%', emotion: 'страх' },
    { date: '15.11.2023', summary: 'Беспокойство о финансах', impact: '-12%', emotion: 'страх' },
    { date: '10.11.2023', summary: 'Страх изменений', impact: '-8%', emotion: 'страх' },
    { date: '05.11.2023', summary: 'Тревога перед встречей', impact: '-10%', emotion: 'страх' },
    { date: '01.11.2023', summary: 'Страх отвержения', impact: '-13%', emotion: 'страх' }
  ],
  surprise: [
    { date: '15.12.2023', summary: 'Неожиданный подарок', impact: '+15%', emotion: 'удивление' },
    { date: '10.12.2023', summary: 'Сюрприз от друзей', impact: '+12%', emotion: 'удивление' },
    { date: '05.12.2023', summary: 'Неожиданная встреча', impact: '+10%', emotion: 'удивление' },
    { date: '01.12.2023', summary: 'Удивительная новость', impact: '+14%', emotion: 'удивление' },
    { date: '25.11.2023', summary: 'Неожиданный успех', impact: '+11%', emotion: 'удивление' },
    { date: '20.11.2023', summary: 'Сюрприз на работе', impact: '+13%', emotion: 'удивление' },
    { date: '15.11.2023', summary: 'Неожиданное приглашение', impact: '+9%', emotion: 'удивление' },
    { date: '10.11.2023', summary: 'Удивительное открытие', impact: '+16%', emotion: 'удивление' },
    { date: '05.11.2023', summary: 'Неожиданный поворот', impact: '+8%', emotion: 'удивление' },
    { date: '01.11.2023', summary: 'Сюрприз от семьи', impact: '+17%', emotion: 'удивление' }
  ],
  sadness: [
    { date: '15.12.2023', summary: 'Грусть от потери', impact: '-15%', emotion: 'грусть' },
    { date: '10.12.2023', summary: 'Печаль по поводу разлуки', impact: '-12%', emotion: 'грусть' },
    { date: '05.12.2023', summary: 'Грустные воспоминания', impact: '-10%', emotion: 'грусть' },
    { date: '01.12.2023', summary: 'Печаль от неудачи', impact: '-14%', emotion: 'грусть' },
    { date: '25.11.2023', summary: 'Грусть в одиночестве', impact: '-11%', emotion: 'грусть' },
    { date: '20.11.2023', summary: 'Печальные новости', impact: '-13%', emotion: 'грусть' },
    { date: '15.11.2023', summary: 'Грусть от расставания', impact: '-9%', emotion: 'грусть' },
    { date: '10.11.2023', summary: 'Печаль по ушедшим', impact: '-16%', emotion: 'грусть' },
    { date: '05.11.2023', summary: 'Грустный день', impact: '-8%', emotion: 'грусть' },
    { date: '01.11.2023', summary: 'Печаль от ошибок', impact: '-17%', emotion: 'грусть' }
  ],
  disgust: [
    { date: '15.12.2023', summary: 'Отвращение от еды', impact: '-15%', emotion: 'отвращение' },
    { date: '10.12.2023', summary: 'Отвращение к поведению', impact: '-12%', emotion: 'отвращение' },
    { date: '05.12.2023', summary: 'Брезгливость от грязи', impact: '-10%', emotion: 'отвращение' },
    { date: '01.12.2023', summary: 'Отвращение к запаху', impact: '-14%', emotion: 'отвращение' },
    { date: '25.11.2023', summary: 'Брезгливость от вида', impact: '-11%', emotion: 'отвращение' },
    { date: '20.11.2023', summary: 'Отвращение к ситуации', impact: '-13%', emotion: 'отвращение' },
    { date: '15.11.2023', summary: 'Брезгливость от мысли', impact: '-9%', emotion: 'отвращение' },
    { date: '10.11.2023', summary: 'Отвращение к человеку', impact: '-16%', emotion: 'отвращение' },
    { date: '05.11.2023', summary: 'Брезгливость от опыта', impact: '-8%', emotion: 'отвращение' },
    { date: '01.11.2023', summary: 'Отвращение к вкусу', impact: '-17%', emotion: 'отвращение' }
  ],
  anger: [
    { date: '15.12.2023', summary: 'Гнев от несправедливости', impact: '-15%', emotion: 'гнев' },
    { date: '10.12.2023', summary: 'Злость на коллегу', impact: '-12%', emotion: 'гнев' },
    { date: '05.12.2023', summary: 'Гнев от задержки', impact: '-10%', emotion: 'гнев' },
    { date: '01.12.2023', summary: 'Злость на ситуацию', impact: '-14%', emotion: 'гнев' },
    { date: '25.11.2023', summary: 'Гнев от обмана', impact: '-11%', emotion: 'гнев' },
    { date: '20.11.2023', summary: 'Злость на себя', impact: '-13%', emotion: 'гнев' },
    { date: '15.11.2023', summary: 'Гнев от ошибки', impact: '-9%', emotion: 'гнев' },
    { date: '10.11.2023', summary: 'Злость на систему', impact: '-16%', emotion: 'гнев' },
    { date: '05.11.2023', summary: 'Гнев от провала', impact: '-8%', emotion: 'гнев' },
    { date: '01.11.2023', summary: 'Злость на обстоятельства', impact: '-17%', emotion: 'гнев' }
  ],
  anticipation: [
    { date: '15.12.2023', summary: 'Ожидание праздника', impact: '+15%', emotion: 'ожидание' },
    { date: '10.12.2023', summary: 'Предвкушение встречи', impact: '+12%', emotion: 'ожидание' },
    { date: '05.12.2023', summary: 'Ожидание новостей', impact: '+10%', emotion: 'ожидание' },
    { date: '01.12.2023', summary: 'Предвкушение поездки', impact: '+14%', emotion: 'ожидание' },
    { date: '25.11.2023', summary: 'Ожидание успеха', impact: '+11%', emotion: 'ожидание' },
    { date: '20.11.2023', summary: 'Предвкушение события', impact: '+13%', emotion: 'ожидание' },
    { date: '15.11.2023', summary: 'Ожидание ответа', impact: '+9%', emotion: 'ожидание' },
    { date: '10.11.2023', summary: 'Предвкушение изменения', impact: '+16%', emotion: 'ожидание' },
    { date: '05.11.2023', summary: 'Ожидание результата', impact: '+8%', emotion: 'ожидание' },
    { date: '01.11.2023', summary: 'Предвкушение приключения', impact: '+17%', emotion: 'ожидание' }
  ]
};

const aiAdvice = {
  joy: 'Практикуйте благодарность каждый день - записывайте 3 вещи, за которые вы благодарны.',
  trust: 'Укрепляйте доверие через открытое общение и выполнение обещаний.',
  fear: 'Разберитесь с источниками страха через медитацию и анализ ситуаций.',
  surprise: 'Позвольте себе больше спонтанности и новых впечатлений.',
  sadness: 'Разрешите себе чувствовать грусть и найдите здоровые способы выражения эмоций.',
  disgust: 'Определите, что вызывает отвращение и установите здоровые границы.',
  anger: 'Научитесь распознавать ранние признаки гнева и используйте техники дыхания.',
  anticipation: 'Ставьте реалистичные цели и разбивайте их на достижимые шаги.'
};

const getGradient = (emotion) => {
  const gradients = {
    joy: 'linear-gradient(90deg, rgba(255, 200, 0, 0.2), rgba(255, 200, 0, 1))',
    sadness: 'linear-gradient(90deg, rgba(40, 140, 220, 0.2), rgba(40, 140, 220, 1))',
    anger: 'linear-gradient(90deg, rgba(230, 60, 50, 0.2), rgba(230, 60, 50, 1))',
    fear: 'linear-gradient(90deg, rgba(150, 80, 190, 0.2), rgba(150, 80, 190, 1))',
    surprise: 'linear-gradient(90deg, rgba(50, 100, 120, 0.2), rgba(80, 140, 150, 1))',
    trust: 'linear-gradient(90deg, rgba(40, 200, 120, 0.2), rgba(40, 200, 120, 1))',
    anticipation: 'linear-gradient(90deg, rgba(80, 120, 160, 0.2), rgba(100, 140, 180, 1))',
    disgust: 'linear-gradient(90deg, rgba(110, 80, 70, 0.2), rgba(110, 80, 70, 1))'
  };
  return gradients[emotion] || 'linear-gradient(90deg, rgba(128, 128, 128, 0.2), rgba(128, 128, 128, 1))';
};

const getEmotionColor = (emotion, opacity = 1) => {
  const colors = {
    joy: `rgba(255, 200, 0, ${opacity})`,
    trust: `rgba(40, 200, 120, ${opacity})`,
    sadness: `rgba(40, 140, 220, ${opacity})`,
    anger: `rgba(230, 60, 50, ${opacity})`,
    fear: `rgba(150, 80, 190, ${opacity})`,
    surprise: `rgba(80, 140, 150, ${opacity})`,
    anticipation: `rgba(100, 140, 180, ${opacity})`,
    disgust: `rgba(110, 80, 70, ${opacity})`
  };
  return colors[emotion] || `rgba(255,0,0,${opacity})`;
};

const getEmotionData = (emotionKey) => {
  return emotionsDataByTime[emotionKey] || {
    day: { value: 0, trend: 'stable' },
    week: { value: 0, trend: 'stable' },
    month: { value: 0, trend: 'stable' },
    all: { value: 0, trend: 'stable' },
    name: emotionKey.charAt(0).toUpperCase() + emotionKey.slice(1),
    emotion: emotionKey
  };
};

const getTopicsByEmotion = (emotionKey) => {
  return topicsByEmotion[emotionKey] || [];
};

const getAIAdvice = (emotionKey) => {
  return aiAdvice[emotionKey] || 'Нет совета для этой эмоции.';
};

export {
  emotionsDataByTime,
  emotionsList,
  emotionToPlanetIndex,
  getGradient,
  getEmotionColor,
  getEmotionData,
  getTopicsByEmotion,
  getAIAdvice
};
