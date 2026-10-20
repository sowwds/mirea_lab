#include "stm32c0xx_hal.h"

// Nucleo-C031C6: выводы соответствуют diagram.json.
#define RED_GPIO_Port       GPIOA
#define RED_Pin             GPIO_PIN_5   // D13
#define YELLOW_GPIO_Port    GPIOA
#define YELLOW_Pin          GPIO_PIN_6   // D12
#define GREEN_GPIO_Port     GPIOA
#define GREEN_Pin           GPIO_PIN_7   // D11, TIM3_CH2 (AF1)
#define PED_RED_GPIO_Port   GPIOC
#define PED_RED_Pin         GPIO_PIN_7   // D9
#define PED_GREEN_GPIO_Port GPIOA
#define PED_GREEN_Pin       GPIO_PIN_9   // D8
#define BUT_GPIO_Port       GPIOB
#define BUT_Pin             GPIO_PIN_10  // D4

#define PWM_PERIOD_TICKS 1000U

TIM_HandleTypeDef htim3;

static void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_TIM3_Init(void);
static void SetGreenBrightness(uint8_t percent);

int main(void) {
  uint8_t brightness = 10;
  int8_t brightness_step = 10;

  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();
  MX_TIM3_Init();
  HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_2);

  while (1) {
    // едут машины: яркость зелёного плавно меняется ШИМ-сигналом TIM3.
    HAL_GPIO_WritePin(RED_GPIO_Port, RED_Pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(YELLOW_GPIO_Port, YELLOW_Pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(PED_RED_GPIO_Port, PED_RED_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(PED_GREEN_GPIO_Port, PED_GREEN_Pin, GPIO_PIN_RESET);

    SetGreenBrightness(brightness);
    HAL_Delay(100);

    if (brightness == 100U) {
      brightness_step = -10;
    } else if (brightness == 10U) {
      brightness_step = 10;
    }
    brightness = (uint8_t)((int16_t)brightness + brightness_step);

    if (HAL_GPIO_ReadPin(BUT_GPIO_Port, BUT_Pin) == GPIO_PIN_SET) {
      // мигание для машин
      for (int i = 0; i < 6; i++) {
        SetGreenBrightness((i % 2 == 0) ? 0U : 100U);
        HAL_Delay(500);
      }

      // машины жёлтый, пеш красный
      SetGreenBrightness(0);
      HAL_GPIO_WritePin(YELLOW_GPIO_Port, YELLOW_Pin, GPIO_PIN_SET);
      HAL_Delay(2000);

      // машины красный, пеш зелёный
      HAL_GPIO_WritePin(YELLOW_GPIO_Port, YELLOW_Pin, GPIO_PIN_RESET);
      HAL_GPIO_WritePin(RED_GPIO_Port, RED_Pin, GPIO_PIN_SET);
      HAL_GPIO_WritePin(PED_RED_GPIO_Port, PED_RED_Pin, GPIO_PIN_RESET);
      HAL_GPIO_WritePin(PED_GREEN_GPIO_Port, PED_GREEN_Pin, GPIO_PIN_SET);
      HAL_Delay(10000);

      // пеш мигает 3 сек
      for (int i = 0; i < 6; i++) {
        HAL_GPIO_TogglePin(PED_GREEN_GPIO_Port, PED_GREEN_Pin);
        HAL_Delay(500);
      }

      // пеш красный, машинам красный и жёлтый
      HAL_GPIO_WritePin(PED_GREEN_GPIO_Port, PED_GREEN_Pin, GPIO_PIN_RESET);
      HAL_GPIO_WritePin(PED_RED_GPIO_Port, PED_RED_Pin, GPIO_PIN_SET);
      HAL_GPIO_WritePin(YELLOW_GPIO_Port, YELLOW_Pin, GPIO_PIN_SET);
      HAL_Delay(2000);

      // повторный цикл
      while (HAL_GPIO_ReadPin(BUT_GPIO_Port, BUT_Pin) == GPIO_PIN_SET) {
        HAL_Delay(10);
      }
    }
  }
}

static void SetGreenBrightness(uint8_t percent) {
  uint32_t pulse = (PWM_PERIOD_TICKS * percent) / 100U;
  __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, pulse);
}

static void MX_TIM3_Init(void) {
  TIM_OC_InitTypeDef channel = {0};

  __HAL_RCC_TIM3_CLK_ENABLE();

  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 48 - 1;              // 48 МГц -> 1 МГц
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = PWM_PERIOD_TICKS - 1;   // 1 МГц / 1000 -> 1 кГц
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_PWM_Init(&htim3) != HAL_OK) {
    while (1) { }
  }

  channel.OCMode = TIM_OCMODE_PWM1;
  channel.Pulse = 0;
  channel.OCPolarity = TIM_OCPOLARITY_HIGH;
  channel.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &channel, TIM_CHANNEL_2) != HAL_OK) {
    while (1) { }
  }
}

static void SystemClock_Config(void) {
  RCC_OscInitTypeDef osc = {0};
  RCC_ClkInitTypeDef clk = {0};

  osc.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  osc.HSIState = RCC_HSI_ON;
  osc.HSIDiv = RCC_HSI_DIV1;
  osc.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  if (HAL_RCC_OscConfig(&osc) != HAL_OK) {
    while (1) { }
  }

  clk.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1;
  clk.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
  clk.AHBCLKDivider = RCC_SYSCLK_DIV1;
  clk.APB1CLKDivider = RCC_HCLK_DIV1;
  if (HAL_RCC_ClockConfig(&clk, FLASH_LATENCY_1) != HAL_OK) {
    while (1) { }
  }
}

static void MX_GPIO_Init(void) {
  GPIO_InitTypeDef gpio = {0};

  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();

  HAL_GPIO_WritePin(GPIOA, RED_Pin | YELLOW_Pin | PED_GREEN_Pin, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(GPIOC, PED_RED_Pin, GPIO_PIN_RESET);

  gpio.Mode = GPIO_MODE_OUTPUT_PP;
  gpio.Pull = GPIO_NOPULL;
  gpio.Speed = GPIO_SPEED_FREQ_LOW;
  gpio.Pin = RED_Pin | YELLOW_Pin | PED_GREEN_Pin;
  HAL_GPIO_Init(GPIOA, &gpio);

  gpio.Pin = PED_RED_Pin;
  HAL_GPIO_Init(GPIOC, &gpio);

  gpio.Mode = GPIO_MODE_AF_PP;
  gpio.Pull = GPIO_NOPULL;
  gpio.Speed = GPIO_SPEED_FREQ_LOW;
  gpio.Alternate = GPIO_AF1_TIM3;
  gpio.Pin = GREEN_Pin;
  HAL_GPIO_Init(GPIOA, &gpio);

  gpio.Mode = GPIO_MODE_INPUT;
  gpio.Pull = GPIO_PULLDOWN;
  gpio.Pin = BUT_Pin;
  HAL_GPIO_Init(GPIOB, &gpio);
}
