# Практическая работа №4

**Тема:** Принцип работы таймера в режиме ШИМ. Регулирование яркости светодиода

## Содержание

## Введение

Широтно-импульсная модуляция применяется для управления средней мощностью нагрузки с помощью цифрового сигнала. Один из наглядных примеров её использования — изменение яркости светодиода за счёт регулирования длительности включённого состояния при постоянной частоте импульсов [1].

Цель работы — изучить режим широтно-импульсной модуляции аппаратного таймера STM32 и реализовать управление яркостью светодиода от положения потенциометра.

## 1 Теоретические сведения

### 1.1 Принцип широтно-импульсной модуляции

Широтно-импульсная модуляция, или ШИМ, представляет собой последовательность импульсов, длительность которых регулируется внутри заданного периода. Основные параметры сигнала — период, частота и коэффициент заполнения. Частота показывает число периодов за секунду, а коэффициент заполнения определяет долю периода, в течение которой выход находится в активном состоянии [1, 2].

```text
f = 1 / T
D = tвкл / T × 100%
```

При управлении светодиодом уменьшение времени включения снижает его среднюю яркость. При достаточно высокой частоте отдельные импульсы визуально не различаются. Нулевое заполнение соответствует выключенному светодиоду, а заполнение 100% — непрерывному включению. Промежуточные значения позволяют регулировать яркость; при этом амплитуда логических уровней на выходе остаётся прежней [1].

### 1.2 Формирование ШИМ аппаратным таймером

Аппаратный таймер содержит предделитель PSC, счётчик CNT, регистр периода ARR и регистры сравнения CCR для каналов. Предделитель задаёт скорость счётчика, а ARR — границу его повторяющегося цикла. При счёте вверх значения изменяются от нуля до ARR включительно, поэтому один период содержит ARR плюс один отсчёт [2].

```text
Fcnt = Ftim / (PSC + 1)
Fpwm = Ftim / ((PSC + 1) × (ARR + 1))
```

В режиме PWM1 при счёте вверх и активной высокой полярности канал включён, пока значение CNT меньше CCR. После достижения значения сравнения выход становится неактивным до начала следующего периода. Изменение CCR регулирует длительность импульса, а значит, заполнение сигнала. При этом PSC и ARR могут оставаться постоянными [2].

```text
D = CCR / (ARR + 1) × 100%
```

Генерация продолжается аппаратно после запуска канала: процессор только обновляет регистр сравнения по мере необходимости. Это позволяет одновременно измерять входной сигнал и менять яркость без программного переключения GPIO на каждом импульсе.

### 1.3 Таймер TIM3 и аналоговое управление

В STM32C031C6 TIM3 относится к 16-разрядным таймерам общего назначения и поддерживает четыре канала. В проекте выбран его второй канал: функция TIM3_CH2 доступна на PA7 при выборе альтернативной функции AF1. Вывод сохраняет цифровую природу, но его состоянием управляет таймер [3].

Для задания яркости используется потенциометр, подключённый между питанием и землёй. Его средний вывод передаёт изменяемое напряжение на аналоговый вход микроконтроллера. АЦП преобразует это напряжение в число; при разрешении 12 бит результат содержит значения от 0 до 4095 [3, 4].

Показание АЦП масштабируется в диапазон регистра сравнения. Таким образом, положение ручки определяет коэффициент заполнения ШИМ, а частота импульсов остаётся постоянной. В данной реализации промежуточное положение потенциометра задаёт устойчивую промежуточную яркость.

## 2 Практическая реализация

### 2.1 Схема подключения

Проект выполнен на C с библиотекой STM32 HAL для платы Nucleo-C031C6. В схеме Wokwi используются один зелёный светодиод, последовательный резистор 220 Ом и потенциометр.

Анод светодиода соединён с D11, соответствующим PA7 и каналу TIM3_CH2. Катод подключён к земле через резистор. Контакты VCC и GND потенциометра соединены соответственно с 3,3 В и землёй. Его выход SIG подключён к A0, соответствующему PA0 и входу ADC_IN0. Полная схема соединений хранится в diagram.json и приведена в Листинге 2.1.

```json
{
  "version": 1,
  "author": "Aleksey",
  "editor": "wokwi",
  "parts": [
    { "type": "board-st-nucleo-c031c6", "id": "nucleo", "top": 0, "left": 0, "attrs": {} },
    { "type": "wokwi-led", "id": "led", "top": 0, "left": 390, "attrs": { "color": "green", "label": "PWM", "gamma": "1.0", "fps": "80" } },
    { "type": "wokwi-resistor", "id": "r1", "top": 65, "left": 380, "rotate": 90, "attrs": { "value": "220" } },
    { "type": "wokwi-potentiometer", "id": "pot", "top": 180, "left": 380, "attrs": { "value": "0" } }
  ],
  "connections": [
    [ "nucleo:D11", "led:A", "green", [] ],
    [ "led:C", "r1:1", "black", [] ],
    [ "r1:2", "nucleo:GND.1", "black", [] ],
    [ "pot:VCC", "nucleo:3V3", "red", [] ],
    [ "pot:GND", "nucleo:GND.1", "black", [] ],
    [ "pot:SIG", "nucleo:A0", "orange", [] ]
  ],
  "dependencies": {}
}
```

Для модели светодиода задано значение gamma, равное 1,0. Оно отключает стандартную гамма-коррекцию отображения Wokwi и делает изменение заполнения более наглядным. Эта настройка влияет на изображение светодиода в симуляторе, а не на параметры аппаратного таймера [5].

### 2.2 Расчёт и настройка TIM3

В программе используется внутренний источник HSI с частотой 48 МГц. Делители AHB и APB1 установлены в единицу, поэтому частота тактирования TIM3 в данной конфигурации составляет 48 МГц.

Предделитель выбран равным 47. После деления на PSC плюс один частота счётчика составляет 1 МГц, то есть один отсчёт занимает 1 мкс. Регистр периода ARR равен 999: счётчик проходит 1000 значений, после чего начинает новый период.

```text
Fcnt = 48 000 000 / (47 + 1) = 1 000 000 Гц
Fpwm = 1 000 000 / (999 + 1) = 1 000 Гц
Tpwm = 1 / 1 000 = 1 мс
```

В функции MX_TIM3_Init включается тактирование TIM3, задаётся счёт вверх и выполняется HAL_TIM_PWM_Init. Для второго канала выбран PWM1 с высокой активной полярностью. Начальное значение Pulse равно нулю, поэтому перед первым обновлением заполнения светодиод выключен.

В MX_GPIO_Init вывод PA7 переведён в режим GPIO_MODE_AF_PP, а параметр GPIO_AF1_TIM3 соединяет его с выходом таймера. После настройки функция HAL_TIM_PWM_Start запускает TIM_CHANNEL_2. Для изменения яркости достаточно обновлять CCR2; повторно запускать канал при каждом измерении не требуется.

### 2.3 Чтение потенциометра и изменение яркости

ADC1 настроен на однократное программно запускаемое измерение канала ADC_CHANNEL_0 с разрешением 12 бит. Вывод PA0 работает в аналоговом режиме. Перед использованием выполняется калибровка АЦП.

В основном цикле HAL_ADC_Start запускает преобразование, HAL_ADC_PollForConversion ожидает его завершения, HAL_ADC_GetValue возвращает результат, а HAL_ADC_Stop завершает текущую операцию. Параметр 100 в функции ожидания задаёт предельное время в миллисекундах; готовое измерение считывается без обязательного ожидания всего этого интервала.

Следующая операция пересчитывает показание АЦП в регистр сравнения второго канала:

```text
CCR2 = 1000 × value / 4095
```

При минимальном значении АЦП CCR2 равен нулю и светодиод выключен. При значении 2048 целочисленный расчёт даёт CCR2, равный 500, что соответствует заполнению 50%. При максимальном значении 4095 получается CCR2, равный 1000: счётчик от нуля до 999 всё время меньше регистра сравнения, поэтому выход непрерывно активен.

Приведение value к uint32_t обеспечивает 32-разрядную арифметику промежуточного произведения. Макрос __HAL_TIM_SET_COMPARE записывает вычисленное значение в CCR2. Пауза HAL_Delay(10) задаёт интервал между опросами потенциометра, а TIM3 во время этой паузы продолжает формировать ШИМ. Автоматического изменения яркости при неподвижной ручке программа не выполняет.

Полный исходный файл main.c, включающий основной цикл, настройку АЦП, таймера, тактирования и GPIO, приведён в Листинге 2.2.

```c
#include "stm32c0xx_hal.h"

#define LED_Pin GPIO_PIN_7       // D11, PA7, TIM3_CH2 (AF1)
#define POT_Pin GPIO_PIN_0       // A0, PA0, ADC_IN0
#define PWM_PERIOD_TICKS 1000U

ADC_HandleTypeDef hadc1;
TIM_HandleTypeDef htim3;

static void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_ADC1_Init(void);
static void MX_TIM3_Init(void);

int main(void) {
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();
  MX_ADC1_Init();
  MX_TIM3_Init();
  HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_2);

  uint16_t value = 0;

  while (1) {
    HAL_ADC_Start(&hadc1);
    HAL_ADC_PollForConversion(&hadc1, 100);
    value = (uint16_t)HAL_ADC_GetValue(&hadc1);
    HAL_ADC_Stop(&hadc1);

    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2,
                          PWM_PERIOD_TICKS * (uint32_t)value / 4095U);
    HAL_Delay(10);
  }
}

static void MX_ADC1_Init(void) {
  ADC_ChannelConfTypeDef channel = {0};
  __HAL_RCC_ADC_CLK_ENABLE();

  hadc1.Instance = ADC1;
  hadc1.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV4;
  hadc1.Init.Resolution = ADC_RESOLUTION_12B;
  hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc1.Init.ScanConvMode = ADC_SCAN_SEQ_FIXED;
  hadc1.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  hadc1.Init.LowPowerAutoWait = DISABLE;
  hadc1.Init.LowPowerAutoPowerOff = DISABLE;
  hadc1.Init.ContinuousConvMode = DISABLE;
  hadc1.Init.NbrOfConversion = 1;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc1.Init.DMAContinuousRequests = DISABLE;
  hadc1.Init.Overrun = ADC_OVR_DATA_PRESERVED;
  hadc1.Init.SamplingTimeCommon1 = ADC_SAMPLETIME_39CYCLES_5;
  hadc1.Init.SamplingTimeCommon2 = ADC_SAMPLETIME_39CYCLES_5;
  hadc1.Init.OversamplingMode = DISABLE;
  HAL_ADC_Init(&hadc1);

  channel.Channel = ADC_CHANNEL_0;
  channel.Rank = ADC_RANK_CHANNEL_NUMBER;
  channel.SamplingTime = ADC_SAMPLINGTIME_COMMON_1;
  HAL_ADC_ConfigChannel(&hadc1, &channel);
  HAL_ADCEx_Calibration_Start(&hadc1);
}

static void MX_TIM3_Init(void) {
  TIM_OC_InitTypeDef channel = {0};
  __HAL_RCC_TIM3_CLK_ENABLE();

  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 48 - 1;             // 48 МГц -> 1 МГц
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = PWM_PERIOD_TICKS - 1;  // 1 МГц / 1000 -> 1 кГц
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  HAL_TIM_PWM_Init(&htim3);

  channel.OCMode = TIM_OCMODE_PWM1;
  channel.Pulse = 0;
  channel.OCPolarity = TIM_OCPOLARITY_HIGH;
  channel.OCFastMode = TIM_OCFAST_DISABLE;
  HAL_TIM_PWM_ConfigChannel(&htim3, &channel, TIM_CHANNEL_2);
}

static void SystemClock_Config(void) {
  RCC_OscInitTypeDef osc = {0};
  RCC_ClkInitTypeDef clk = {0};
  osc.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  osc.HSIState = RCC_HSI_ON;
  osc.HSIDiv = RCC_HSI_DIV1;
  osc.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  HAL_RCC_OscConfig(&osc);
  clk.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1;
  clk.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
  clk.AHBCLKDivider = RCC_SYSCLK_DIV1;
  clk.APB1CLKDivider = RCC_HCLK_DIV1;
  HAL_RCC_ClockConfig(&clk, FLASH_LATENCY_1);
}

static void MX_GPIO_Init(void) {
  GPIO_InitTypeDef gpio = {0};
  __HAL_RCC_GPIOA_CLK_ENABLE();

  gpio.Pin = LED_Pin;
  gpio.Mode = GPIO_MODE_AF_PP;
  gpio.Pull = GPIO_NOPULL;
  gpio.Speed = GPIO_SPEED_FREQ_LOW;
  gpio.Alternate = GPIO_AF1_TIM3;
  HAL_GPIO_Init(GPIOA, &gpio);

  gpio.Pin = POT_Pin;
  gpio.Mode = GPIO_MODE_ANALOG;
  gpio.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOA, &gpio);
}
```

### 2.4 Сборка и проверка реализации

Локальная сборка выполняется скриптом build-local.sh через Arduino CLI с ядром STM32Duino и выбранной платой NUCLEO_C031C6. Скрипт формирует файлы prac4.ino.bin и prac4.ino.elf в каталоге .build. Файл wokwi.toml указывает симулятору на эти результаты сборки. Текущая версия программы успешно скомпилирована: объём кода составляет 11 880 байт, статических данных — 968 байт.

Для проверки в Wokwi необходимо установить ручку потенциометра сначала в минимум, затем в середину и максимум диапазона. Эти положения соответствуют выключенному светодиоду, промежуточной яркости и максимальной яркости. При удержании ручки на месте программа сохраняет выбранное заполнение. Изменение положения ручки должно менять яркость без автоматической пульсации.

## Заключение

В ходе практической работы рассмотрен принцип широтно-импульсной модуляции и рассчитаны параметры аппаратного таймера TIM3 для формирования сигнала частотой 1 кГц.

Реализовано прямое управление яркостью светодиода от потенциометра. ADC1 измеряет входное напряжение, а программа масштабирует результат в диапазон CCR2 от 0 до 1000. Изменение заполнения от 0 до 100% регулирует яркость при постоянной частоте ШИМ. Генерацию импульсов выполняет таймер, а основной цикл занимается измерением и обновлением регистра сравнения.

## Список использованных источников

1. Таймеры STM32: управление временем в микроконтроллере, примеры [Электронный ресурс]. — Skypro Wiki. — URL: https://sky.pro/wiki/gamedev/ispolzovanie-tajmerov-na-stm32/ (дата обращения: 10.11.2026).
2. STM32C0 series advanced Arm-based 32-bit MCUs. RM0490 [Электронный ресурс]. — STMicroelectronics. — URL: https://www.st.com/resource/en/reference_manual/rm0490-stm32c0-series-advanced-armbased-32bit-mcus-stmicroelectronics.pdf (дата обращения: 10.11.2026).
3. STM32C031x4/x6. DS13867 [Электронный ресурс]. — STMicroelectronics. — URL: https://www.st.com/resource/en/datasheet/stm32c031c6.pdf (дата обращения: 10.11.2026).
4. Wokwi-potentiometer Reference [Электронный ресурс]. — Wokwi Docs. — URL: https://docs.wokwi.com/parts/wokwi-potentiometer (дата обращения: 10.11.2026).
5. Wokwi-led Reference [Электронный ресурс]. — Wokwi Docs. — URL: https://docs.wokwi.com/parts/wokwi-led (дата обращения: 10.11.2026).

