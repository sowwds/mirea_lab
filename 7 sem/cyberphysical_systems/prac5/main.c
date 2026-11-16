#include "stm32c0xx_hal.h"

#define SERVO_Pin GPIO_PIN_7     // D11 / PA7 / TIM3_CH2 (AF1)
#define POT_Pin GPIO_PIN_0       // A0 / PA0 / ADC_IN0
#define CW_Pin GPIO_PIN_10       // D2 / PA10, button to GND
#define CCW_Pin GPIO_PIN_3       // D3 / PB3, button to GND
#define STEP_Pin GPIO_PIN_5      // D6 / PB5 -> A4988 STEP
#define DIR_Pin GPIO_PIN_4       // D5 / PB4 -> A4988 DIR
#define SERVO_PERIOD_US 20000U   // 50 Hz
#define SERVO_MIN_US 500U
#define SERVO_MAX_US 2500U
#define STEP_HALF_PERIOD_MS 2U   // 250 steps/s, 200 steps/revolution

ADC_HandleTypeDef hadc1;
TIM_HandleTypeDef htim3;

static void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_ADC1_Init(void);
static void MX_TIM3_Init(void);

/* Change direction with STEP low, then wait before the next pulse. */
static void Stepper_Update(uint32_t now) {
  static uint32_t changed_at = 0;
  static GPIO_PinState step = GPIO_PIN_RESET;
  static GPIO_PinState direction = GPIO_PIN_RESET;
  uint8_t cw = HAL_GPIO_ReadPin(GPIOA, CW_Pin) == GPIO_PIN_RESET;
  uint8_t ccw = HAL_GPIO_ReadPin(GPIOB, CCW_Pin) == GPIO_PIN_RESET;

  /* Both released or both pressed: stop. */
  if (cw == ccw) {
    step = GPIO_PIN_RESET;
    HAL_GPIO_WritePin(GPIOB, STEP_Pin, step);
    changed_at = now;
    return;
  }

  GPIO_PinState requested = cw ? GPIO_PIN_SET : GPIO_PIN_RESET;
  if (requested != direction) {
    step = GPIO_PIN_RESET;
    HAL_GPIO_WritePin(GPIOB, STEP_Pin, step);
    direction = requested;
    HAL_GPIO_WritePin(GPIOB, DIR_Pin, direction);
    changed_at = now;
    return;
  }

  if ((uint32_t)(now - changed_at) >= STEP_HALF_PERIOD_MS) {
    changed_at = now;
    step = (step == GPIO_PIN_RESET) ? GPIO_PIN_SET : GPIO_PIN_RESET;
    HAL_GPIO_WritePin(GPIOB, STEP_Pin, step);
  }
}

int main(void) {
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();
  MX_ADC1_Init();
  MX_TIM3_Init();
  HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_2);

  uint32_t sampled_at = HAL_GetTick() - 20U;
  while (1) {
    uint32_t now = HAL_GetTick();
    if ((uint32_t)(now - sampled_at) >= 20U) {
      sampled_at = now;
      HAL_ADC_Start(&hadc1);
      if (HAL_ADC_PollForConversion(&hadc1, 2U) == HAL_OK) {
        uint32_t value = HAL_ADC_GetValue(&hadc1);
        uint32_t pulse = SERVO_MIN_US +
            (SERVO_MAX_US - SERVO_MIN_US) * value / 4095U;
        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, pulse);
      }
      HAL_ADC_Stop(&hadc1);
    }

    Stepper_Update(HAL_GetTick());
    HAL_Delay(1U);
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
  htim3.Init.Period = SERVO_PERIOD_US - 1;  // 1 MHz / 20000 -> 50 Hz
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  HAL_TIM_PWM_Init(&htim3);

  channel.OCMode = TIM_OCMODE_PWM1;
  channel.Pulse = 1500;                  // Initial position: 90 degrees
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
  __HAL_RCC_GPIOB_CLK_ENABLE();

  gpio.Pin = SERVO_Pin;
  gpio.Mode = GPIO_MODE_AF_PP;
  gpio.Pull = GPIO_NOPULL;
  gpio.Speed = GPIO_SPEED_FREQ_LOW;
  gpio.Alternate = GPIO_AF1_TIM3;
  HAL_GPIO_Init(GPIOA, &gpio);

  gpio.Pin = POT_Pin;
  gpio.Mode = GPIO_MODE_ANALOG;
  gpio.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOA, &gpio);

  gpio.Pin = CW_Pin;
  gpio.Mode = GPIO_MODE_INPUT;
  gpio.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(GPIOA, &gpio);
  gpio.Pin = CCW_Pin;
  HAL_GPIO_Init(GPIOB, &gpio);

  HAL_GPIO_WritePin(GPIOB, STEP_Pin | DIR_Pin, GPIO_PIN_RESET);
  gpio.Pin = STEP_Pin | DIR_Pin;
  gpio.Mode = GPIO_MODE_OUTPUT_PP;
  gpio.Pull = GPIO_NOPULL;
  gpio.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &gpio);
}
