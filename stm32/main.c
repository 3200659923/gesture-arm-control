/*==============================================================================
 * main.c - 第三步：PCA9685 + 舵机控制
 *------------------------------------------------------------------------------
 * 功能：串口发 Axx,Bxx,Cxx,Dxx\n → 解析 → 4 路舵机转到对应角度 → 回显
 * 硬件：STM32F103C8T6 + PCA9685(I2C) + 4 路舵机
 * 接线：
 *   USART1: PA9=TX, PA10=RX  (USB-TTL)
 *   I2C:    PB6=SCL, PB7=SDA  (PCA9685)
 *   PCA9685 VCC=3.3V, V+=5V独立电源, GND共地
 *   舵机: CH0=底座(A), CH1=大臂(B), CH2=小臂(C), CH3=夹爪(D)
 *============================================================================*/
#include "stm32f10x.h"

/*============================ 串口发送 ============================*/
static void uart_send(const char *s)
{
    while (*s) {
        while (!(USART1->SR & (1UL << 7)));  /* TXE */
        USART1->DR = (uint8_t)(*s++);
    }
}

static void uart_send_num(uint8_t num)
{
    char buf[4];
    uint8_t i = 0;
    if (num == 0) {
        while (!(USART1->SR & (1UL << 7)));
        USART1->DR = '0';
        return;
    }
    while (num > 0) {
        buf[i++] = '0' + (num % 10);
        num /= 10;
    }
    while (i > 0) {
        while (!(USART1->SR & (1UL << 7)));
        USART1->DR = buf[--i];
    }
}

/*============================ 软件 I2C ============================*/
#define I2C_SCL_PIN  (1UL << 6)   /* PB6 */
#define I2C_SDA_PIN  (1UL << 7)   /* PB7 */

#define SCL_H()   (GPIOB->BSRR = I2C_SCL_PIN)
#define SCL_L()   (GPIOB->BRR  = I2C_SCL_PIN)
#define SDA_H()   (GPIOB->BSRR = I2C_SDA_PIN)
#define SDA_L()   (GPIOB->BRR  = I2C_SDA_PIN)
#define SDA_READ() ((GPIOB->IDR & I2C_SDA_PIN) ? 1 : 0)

static void i2c_delay(void)
{
    volatile uint32_t i;
    for (i = 0; i < 10; i++);   /* 约 5us，100kHz */
}

static void i2c_init(void)
{
    RCC->APB2ENR |= (1UL << 3);   /* GPIOB 时钟 */
    GPIOB->ODR |= (I2C_SCL_PIN | I2C_SDA_PIN);   /* 上拉 */
    /* PB6/PB7 开漏输出 10MHz */
    GPIOB->CRL &= ~((0xFUL << 24) | (0xFUL << 28));
    GPIOB->CRL |=  ((0x5UL << 24) | (0x5UL << 28));
}

static void i2c_start(void)
{
    SDA_H(); SCL_H(); i2c_delay();
    SDA_L(); i2c_delay();
    SCL_L();
}

static void i2c_stop(void)
{
    SDA_L(); SCL_H(); i2c_delay();
    SDA_H(); i2c_delay();
}

static uint8_t i2c_write_byte(uint8_t data)
{
    uint8_t i, ack;
    for (i = 0; i < 8; i++) {
        if (data & 0x80) SDA_H();
        else             SDA_L();
        data <<= 1;
        i2c_delay();
        SCL_H(); i2c_delay();
        SCL_L(); i2c_delay();
    }
    SDA_H(); i2c_delay();
    SCL_H(); i2c_delay();
    ack = SDA_READ();   /* 0=ACK */
    SCL_L(); i2c_delay();
    return ack;
}

static void i2c_write_reg(uint8_t dev, uint8_t reg, uint8_t val)
{
    i2c_start();
    i2c_write_byte((dev << 1) | 0x00);
    i2c_write_byte(reg);
    i2c_write_byte(val);
    i2c_stop();
}

/*============================ PCA9685 驱动 ============================*/
#define PCA9685_ADDR      0x40
#define PCA9685_MODE1     0x00
#define PCA9685_PRE_SCALE 0xFE
#define PCA9685_LED0_ON_L 0x06

static void pca9685_init(uint16_t freq_hz)
{
    uint8_t prescale;
    uint32_t tmp;

    /* 计算预分频: prescale = round(25MHz/(4096*freq)) - 1 */
    tmp = 25000000UL / (4096UL * (uint32_t)freq_hz);
    if (tmp > 0) tmp -= 1;
    prescale = (uint8_t)tmp;

    /* 进入 SLEEP，写 PRE_SCALE，唤醒 */
    i2c_write_reg(PCA9685_ADDR, PCA9685_MODE1, 0x10);
    i2c_write_reg(PCA9685_ADDR, PCA9685_PRE_SCALE, prescale);
    i2c_write_reg(PCA9685_ADDR, PCA9685_MODE1, 0x20);   /* 唤醒 + Auto-Increment */
    i2c_write_reg(PCA9685_ADDR, 0x01, 0x04);             /* MODE2: 推挽输出 */
}

static void pca9685_set_angle(uint8_t ch, uint8_t angle)
{
    uint32_t pulse_us, ticks;
    uint8_t reg;

    if (ch > 15) return;
    if (angle > 180) angle = 180;

    /* 角度 -> 脉宽 0.5ms~2.5ms -> 计数 */
    pulse_us = 500UL + (uint32_t)angle * 2000UL / 180UL;
    ticks    = pulse_us * 4096UL / 20000UL;

    reg = PCA9685_LED0_ON_L + ch * 4;
    /* 连续写 4 字节: ON_L, ON_H, OFF_L, OFF_H */
    i2c_start();
    i2c_write_byte((PCA9685_ADDR << 1) | 0x00);
    i2c_write_byte(reg);
    i2c_write_byte(0x00);                    /* ON_L */
    i2c_write_byte(0x00);                    /* ON_H */
    i2c_write_byte((uint8_t)(ticks & 0xFF));/* OFF_L */
    i2c_write_byte((uint8_t)((ticks >> 8) & 0x0F)); /* OFF_H */
    i2c_stop();
}

/*============================ 协议解析 ============================*/
static uint8_t parse_frame(const char *buf, uint8_t *a, uint8_t *b,
                            uint8_t *c, uint8_t *d)
{
    const char *p = buf;
    int val;

    while (*p && *p != 'A') p++;
    if (*p != 'A') return 0;
    p++; val = 0;
    while (*p >= '0' && *p <= '9') { val = val*10 + (*p-'0'); p++; }
    *a = (uint8_t)val;

    while (*p && *p != 'B') p++;
    if (*p != 'B') return 0;
    p++; val = 0;
    while (*p >= '0' && *p <= '9') { val = val*10 + (*p-'0'); p++; }
    *b = (uint8_t)val;

    while (*p && *p != 'C') p++;
    if (*p != 'C') return 0;
    p++; val = 0;
    while (*p >= '0' && *p <= '9') { val = val*10 + (*p-'0'); p++; }
    *c = (uint8_t)val;

    while (*p && *p != 'D') p++;
    if (*p != 'D') return 0;
    p++; val = 0;
    while (*p >= '0' && *p <= '9') { val = val*10 + (*p-'0'); p++; }
    *d = (uint8_t)val;

    return 1;
}

/*============================ 舵机角度限幅 ============================
 * 根据你的机械臂实际结构调整：发角度指令时会自动限制在这个范围内，
 * 防止舵机转到 0°/180° 撞到支架。
 *====================================================================*/
#define SERVO_A_MIN   10    /* 底座最小角度 */
#define SERVO_A_MAX  170    /* 底座最大角度 */
#define SERVO_B_MIN   10    /* 大臂最小角度（防止下垂撞支架）*/
#define SERVO_B_MAX  150    /* 大臂最大角度（防止后仰撞支架）*/
#define SERVO_C_MIN   0    /* 小臂最小角度 */
#define SERVO_C_MAX  120    /* 小臂最大角度（避开135°重力矩最大抖动点）*/
#define SERVO_D_MIN    0    /* 夹爪最小角度（0°=完全张开）*/
#define SERVO_D_MAX  90    /* 夹爪最大角度（90°=完全闭合，实测标定）*/

static uint8_t clamp_angle(uint8_t val, uint8_t min, uint8_t max)
{
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

/*============================ 主函数 ============================*/
int main(void)
{
    char rx_buf[64];
    uint8_t rx_idx = 0;
    uint32_t led_timer = 0;
    uint8_t led_on = 0;
    uint8_t a = 90, b = 90, c = 60, d = 40;   /* 初始角度 */

    /* ----- 时钟 72MHz ----- */
    RCC->CR |= RCC_CR_HSEON;
    while (!(RCC->CR & RCC_CR_HSERDY));
    FLASH->ACR = FLASH_ACR_LATENCY_2 | FLASH_ACR_PRFTBE;
    RCC->CFGR = (1UL << 16) | (7UL << 18) | (4UL << 8);
    RCC->CR |= RCC_CR_PLLON;
    while (!(RCC->CR & RCC_CR_PLLRDY));
    RCC->CFGR |= RCC_CFGR_SW_PLL;
    while ((RCC->CFGR & (3UL << 2)) != (2UL << 2));

    /* ----- GPIOA, GPIOC, USART1 时钟 ----- */
    RCC->APB2ENR |= (1UL << 2) | (1UL << 4) | (1UL << 14);

    /* ----- PA9=TX(复用推挽), PA10=RX(浮空) ----- */
    GPIOA->CRH &= ~(0xFUL << 4);
    GPIOA->CRH |=  (0xBUL << 4);
    GPIOA->CRH &= ~(0xFUL << 8);
    GPIOA->CRH |=  (0x4UL << 8);

    /* ----- PC13=LED ----- */
    GPIOC->CRH &= ~(0xFUL << 20);
    GPIOC->CRH |=  (0x3UL << 20);

    /* ----- USART1 115200 ----- */
    USART1->BRR = 625;
    USART1->CR1 = (1UL << 13) | (1UL << 3) | (1UL << 2);

    /* ----- I2C 初始化 ----- */
    i2c_init();

    /* ----- PCA9685 初始化 50Hz ----- */
    pca9685_init(50);

    /* ----- 所有舵机转到初始角度（先输出 PWM，再上电不会乱转）----- */
    pca9685_set_angle(0, a);   /* CH0 底座 */
    pca9685_set_angle(1, b);   /* CH1 大臂 */
    pca9685_set_angle(2, c);   /* CH2 小臂 */
    pca9685_set_angle(3, d);   /* CH3 夹爪 */

    /* ----- 上电提示 ----- */
    uart_send("\r\n=== ARM Control Ready ===\r\n");
    uart_send("PCA9685 OK, servos at initial position.\r\n");
    uart_send("Now plug in 5V servo power.\r\n");
    uart_send("Send: A90,B90,C60,D40 + 新行\r\n");

    /* ----- 主循环 ----- */
    while (1) {
        /* 接收解析 */
        if (USART1->SR & (1UL << 5)) {   /* RXNE */
            char ch = (char)USART1->DR;
            if (ch == '\n' || ch == '\r') {
                if (rx_idx > 0) {
                    uint8_t na, nb, nc, nd;
                    rx_buf[rx_idx] = '\0';
                    if (parse_frame(rx_buf, &na, &nb, &nc, &nd)) {
                        /* 角度限幅，防止撞支架 */
                        a = clamp_angle(na, SERVO_A_MIN, SERVO_A_MAX);
                        b = clamp_angle(nb, SERVO_B_MIN, SERVO_B_MAX);
                        c = clamp_angle(nc, SERVO_C_MIN, SERVO_C_MAX);
                        d = clamp_angle(nd, SERVO_D_MIN, SERVO_D_MAX);
                        pca9685_set_angle(0, a);
                        pca9685_set_angle(1, b);
                        pca9685_set_angle(2, c);
                        pca9685_set_angle(3, d);
                        uart_send("A="); uart_send_num(a);
                        uart_send(" B="); uart_send_num(b);
                        uart_send(" C="); uart_send_num(c);
                        uart_send(" D="); uart_send_num(d);
                        uart_send("\r\n");
                    } else {
                        uart_send("ERR\r\n");
                    }
                    rx_idx = 0;
                }
            } else if (rx_idx < 63) {
                rx_buf[rx_idx++] = ch;
            }
        }

        /* LED 闪烁 */
        if (++led_timer >= 50000) {
            led_timer = 0;
            led_on = !led_on;
            if (led_on) GPIOC->BRR  = (1UL << 13);
            else        GPIOC->BSRR = (1UL << 13);
        }
    }
}
