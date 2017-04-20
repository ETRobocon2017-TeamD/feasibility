CMD_MAX = 100.0  # 前進/旋回命令絶対最大値
DEG2RAD = 0.01745329238  # 角度単位変換係数(=pi/180)
# EXEC_PERIOD = 0.00400000019  # バランス制御実行周期(秒)
EXEC_PERIOD = 0.050  # バランス制御実行周期(秒)

A_D = 0.8  # ローパスフィルタ係数(左右車輪の平均回転角度用)
A_R = 0.996  # ローパスフィルタ係数(左右車輪の目標平均回転角度用)

# 状態フィードバック係数
# K_F[0]: 車輪回転角度係数
# K_F[1]: 車体傾斜角度係数
# K_F[2]: 車輪回転角速度係数
# K_F[3]: 車体傾斜角速度係数
K_F = [-0.870303, -31.9978, -1.1566 * 0.6, -2.78873]
K_I = -0.44721  # サーボ制御用積分フィードバック係数

K_PHIDOT = 25.0 * 2.5  # 車体目標旋回角速度係数
K_THETADOT = 7.5  # モータ目標回転角速度係数

BATTERY_GAIN = 0.001089  # PWM出力算出用バッテリ電圧補正係数
BATTERY_OFFSET = 0.625  # PWM出力算出用バッテリ電圧補正オフセット

ud_err_theta = 0.0  # 左右車輪の平均回転角度(θ)目標誤差状態値
ud_psi = 0.0  # 車体ピッチ角度(ψ)状態値
ud_theta_lpf = 0.0  # 左右車輪の平均回転角度(θ)状態値
ud_theta_ref = 0.0  # 左右車輪の目標平均回転角度(θ)状態値
ud_thetadot_cmd_lpf = 0.0  # 左右車輪の目標平均回転角速度(dθ/dt)状態値


def rt_saturate(sig, ll, ul):
    if sig >= ul:
        return ul
    elif sig <= ll:
        return ll
    else:
        return sig


def balance_control(args_cmd_forward, args_cmd_turn, args_gyro, args_gyro_offset, args_theta_m_l, args_theta_m_r,
                    args_battery):
    u"""NXTway-GSバランス制御関数。
        この関数は4msec周期で起動されることを前提に設計されています。
        なお、ジャイロセンサオフセット値はセンサ個体および通電によるドリフト
        を伴いますので、適宜補正する必要があります。また、左右の車輪駆動
        モータは個体差により、同じPWM出力を与えても回転数が異なる場合が
        あります。その場合は別途補正機能を追加する必要があります。

    Args:
        args_cmd_forward : 前進/後進命令。100(前進最大値)～-100(後進最大値)
        args_cmd_turn    : 旋回命令。100(右旋回最大値)～-100(左旋回最大値)
        args_gyro        : ジャイロセンサ値
        args_gyro_offset : ジャイロセンサオフセット値
        args_theta_m_l   : 左モータエンコーダ値
        args_theta_m_r   : 右モータエンコーダ値
        args_battery     : バッテリ電圧値(mV)

    Returns:
        (tuple): (左モータPWM出力値, 右モータPWM出力値)
    """
    global ud_err_theta, ud_psi, ud_theta_lpf, ud_theta_ref, ud_thetadot_cmd_lpf
    # print(ud_err_theta, ud_psi, ud_theta_lpf, ud_theta_ref, ud_thetadot_cmd_lpf)
    tmp = [0, 0, 0, 0]
    tmp_theta_0 = [0, 0, 0, 0]

    tmp_thetadot_cmd_lpf = (((args_cmd_forward / CMD_MAX) * K_THETADOT) * (1.0 - A_R)) + (A_R * ud_thetadot_cmd_lpf)
    tmp_theta = (((DEG2RAD * args_theta_m_l) + ud_psi) + ((DEG2RAD * args_theta_m_r) + ud_psi)) * 0.5
    tmp_theta_lpf = ((1.0 - A_D) * tmp_theta) + (A_D * ud_theta_lpf)
    tmp_psidot = (args_gyro - args_gyro_offset) * DEG2RAD
    tmp[0] = ud_theta_ref
    tmp[1] = 0.0
    tmp[2] = tmp_thetadot_cmd_lpf
    tmp[3] = 0.0
    tmp_theta_0[0] = tmp_theta
    tmp_theta_0[1] = ud_psi
    tmp_theta_0[2] = (tmp_theta_lpf - ud_theta_lpf) / EXEC_PERIOD
    tmp_theta_0[3] = tmp_psidot
    tmp_pwm_r_limiter = 0.0
    for tmp_0 in range(4):
        tmp_pwm_r_limiter += (tmp[tmp_0] - tmp_theta_0[tmp_0]) * K_F[tmp_0]

    tmp_pwm_r_limiter = (((K_I * ud_err_theta) + tmp_pwm_r_limiter) /
                         ((BATTERY_GAIN * args_battery) - BATTERY_OFFSET)) * 100

    tmp_pwm_turn = (args_cmd_turn / CMD_MAX) * K_PHIDOT

    tmp_pwm_l_limiter = tmp_pwm_r_limiter + tmp_pwm_turn
    tmp_pwm_l_limiter = rt_saturate(tmp_pwm_l_limiter, -100, 100)
    ret_pwm_l = tmp_pwm_l_limiter

    tmp_pwm_r_limiter -= tmp_pwm_turn
    tmp_pwm_r_limiter = rt_saturate(tmp_pwm_r_limiter, -100, 100)
    ret_pwm_r = tmp_pwm_r_limiter

    tmp_pwm_l_limiter = (EXEC_PERIOD * tmp_thetadot_cmd_lpf) + ud_theta_ref
    tmp_pwm_turn = (EXEC_PERIOD * tmp_psidot) + ud_psi
    tmp_pwm_r_limiter = ((ud_theta_ref - tmp_theta) * EXEC_PERIOD) + ud_err_theta

    ud_err_theta = tmp_pwm_r_limiter
    ud_theta_ref = tmp_pwm_l_limiter
    ud_thetadot_cmd_lpf = tmp_thetadot_cmd_lpf
    ud_psi = tmp_pwm_turn
    ud_theta_lpf = tmp_theta_lpf

    return ret_pwm_l, ret_pwm_r


def balance_init():
    global ud_err_theta, ud_psi, ud_theta_lpf, ud_theta_ref, ud_thetadot_cmd_lpf
    ud_err_theta = 0.0
    ud_theta_ref = 0.0
    ud_thetadot_cmd_lpf = 0.0
    ud_psi = 0.0
    ud_theta_lpf = 0.0
