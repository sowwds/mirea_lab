#include "stm32c0xx_hal.h"

#define RED_GPIO_Port       GPIOA
#define RED_Pin             GPIO_PIN_5   // D13
#define YELLOW_GPIO_Port    GPIOA
#define YELLOW_Pin          GPIO_PIN_6   // D12
#define GREEN_GPIO_Port     GPIOA
#define GREEN_Pin           GPIO_PIN_7   // D11
#define PED_RED_GPIO_Port   GPIOC
#define PED_RED_Pin         GPIO_PIN_7   // D9
#define PED_GREEN_GPIO_Port GPIOA
#define PED_GREEN_Pin       GPIO_PIN_9   // D8
#define BUT_GPIO_Port       GPIOB
#define BUT_Pin             GPIO_PIN_10  // D4

static void SystemClock_Config(void);
static void MX_GPIO_Init(void);

int main(void) {
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();

  while (1) {
    // едут машины
    HAL_GPIO_WritePin(RED_GPIO_Port, RED_Pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(YELLOW_GPIO_Port, YELLOW_Pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(GREEN_GPIO_Port, GREEN_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(PED_RED_GPIO_Port, PED_RED_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(PED_GREEN_GPIO_Port, PED_GREEN_Pin, GPIO_PIN_RESET);

    if (HAL_GPIO_ReadPin(BUT_GPIO_Port, BUT_Pin) == GPIO_PIN_SET) {
      // мигание для машин
      for (int i = 0; i < 6; i++) {
        HAL_GPIO_TogglePin(GREEN_GPIO_Port, GREEN_Pin);
        HAL_Delay(500);
      }

      // машины жёлтый, пеш красный
      HAL_GPIO_WritePin(GREEN_GPIO_Port, GREEN_Pin, GPIO_PIN_RESET);
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

  HAL_GPIO_WritePin(GPIOA, RED_Pin | YELLOW_Pin | GREEN_Pin | PED_GREEN_Pin, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(GPIOC, PED_RED_Pin, GPIO_PIN_RESET);

  gpio.Mode = GPIO_MODE_OUTPUT_PP;
  gpio.Pull = GPIO_NOPULL;
  gpio.Speed = GPIO_SPEED_FREQ_LOW;
  gpio.Pin = RED_Pin | YELLOW_Pin | GREEN_Pin | PED_GREEN_Pin;
  HAL_GPIO_Init(GPIOA, &gpio);

  gpio.Pin = PED_RED_Pin;
  HAL_GPIO_Init(GPIOC, &gpio);

  gpio.Mode = GPIO_MODE_INPUT;
  gpio.Pull = GPIO_PULLDOWN;
  gpio.Pin = BUT_Pin;
  HAL_GPIO_Init(GPIOB, &gpio);
}
