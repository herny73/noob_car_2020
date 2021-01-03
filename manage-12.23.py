#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car

Usage:
    manage.py (drive) [--model=<model>] [--model_1=<model_1>] [--js] [--type=(linear|categorical|rnn|imu|behavior|3d|localizer|latent)] [--camera=(single|stereo)] [--meta=<key:value> ...]
    manage.py (train) [--tub=<tub1,tub2,..tubn>] [--file=<file> ...] (--model=<model>) [--transfer=<model>] [--type=(linear|categorical|rnn|imu|behavior|3d|localizer)] [--continuous] [--aug]


Options:
    -h --help          Show this screen.
    --js               Use physical joystick.
    -f --file=<file>   A text file containing paths to tub files, one per line. Option may be used more than once.
    --meta=<key:value> Key/Value strings describing describing a piece of meta data about this drive. Option may be used more than once.
"""

"""
zmx关于增加模型切换的一些想法
在主进程增加一个变量
       增加一个检测组件
       增加一个tub
在服务器增加个按钮
返回数据中增加变量：是否反转
在网页控制中 显示当前状态序号
"""


import os
import time
import random

from docopt import docopt
import numpy as np

import donkeycar as dk

#import parts
from donkeycar.parts.transform import Lambda, TriggeredCallback, DelayedTrigger
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.controller import LocalWebController, JoystickController
from donkeycar.parts.throttle_filter import ThrottleFilter
from donkeycar.parts.behavior import BehaviorPart
from donkeycar.parts.file_watcher import FileWatcher
from donkeycar.parts.launch import AiLaunch

def drive(cfg, model_path=None, model_path_1=None, use_joystick=False, model_type=None, camera_type='single', meta=[] ):
    '''
    Construct a working robotic vehicle from many parts.
    Each part runs as a job in the Vehicle loop, calling either
    it's run or run_threaded method depending on the constructor flag `threaded`.
    All parts are updated one after another at the framerate given in
    cfg.DRIVE_LOOP_HZ assuming each part finishes processing in a timely manner.
    Parts may have named outputs and inputs. The framework handles passing named outputs
    to parts requesting the same named input.
    '''

    if cfg.DONKEY_GYM:
        #the simulator will use cuda and then we usually run out of resources
        #if we also try to use cuda. so disable for donkey_gym.
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    if model_type is None:
        if cfg.TRAIN_LOCALIZER:
            model_type = "localizer"
        elif cfg.TRAIN_BEHAVIORS:
            model_type = "behavior"
        else:
            model_type = cfg.DEFAULT_MODEL_TYPE

    #Initialize car
    V = dk.vehicle.Vehicle()

    if camera_type == "stereo":

        if cfg.CAMERA_TYPE == "WEBCAM":
            from donkeycar.parts.camera import Webcam

            camA = Webcam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, iCam = 0)
            camB = Webcam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, iCam = 1)

        elif cfg.CAMERA_TYPE == "CVCAM":
            from donkeycar.parts.cv import CvCam

            camA = CvCam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, iCam = 0)
            camB = CvCam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, iCam = 1)
        else:
            raise(Exception("Unsupported camera type: %s" % cfg.CAMERA_TYPE))

        V.add(camA, outputs=['cam/image_array_a'], threaded=True)
        V.add(camB, outputs=['cam/image_array_b'], threaded=True)

        from donkeycar.parts.image import StereoPair

        V.add(StereoPair(), inputs=['cam/image_array_a', 'cam/image_array_b'],
            outputs=['cam/image_array'])

    else:

        inputs = []
        threaded = True
        print("cfg.CAMERA_TYPE", cfg.CAMERA_TYPE)
        if cfg.DONKEY_GYM:
            from donkeycar.parts.dgym import DonkeyGymEnv
            cam = DonkeyGymEnv(cfg.DONKEY_SIM_PATH, env_name=cfg.DONKEY_GYM_ENV_NAME)
            threaded = True
            inputs = ['angle', 'throttle']
        elif cfg.CAMERA_TYPE == "PICAM":
            from donkeycar.parts.camera import PiCamera
            cam = PiCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)
        elif cfg.CAMERA_TYPE == "WEBCAM":
            from donkeycar.parts.camera import Webcam
            cam = Webcam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)
        elif cfg.CAMERA_TYPE == "CVCAM":
            from donkeycar.parts.cv import CvCam
            cam = CvCam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)
        elif cfg.CAMERA_TYPE == "CSIC":
            from donkeycar.parts.camera import CSICamera
            cam = CSICamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, framerate=cfg.CAMERA_FRAMERATE)
        elif cfg.CAMERA_TYPE == "V4L":
            from donkeycar.parts.camera import V4LCamera
            cam = V4LCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, framerate=cfg.CAMERA_FRAMERATE)
        elif cfg.CAMERA_TYPE == "MOCK":
            from donkeycar.parts.camera import MockCamera
            cam = MockCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)
        else:
            raise(Exception("Unkown camera type: %s" % cfg.CAMERA_TYPE))

        V.add(cam, inputs=inputs, outputs=['cam/image_array'], threaded=threaded)

    if use_joystick or cfg.USE_JOYSTICK_AS_DEFAULT:
        #modify max_throttle closer to 1.0 to have more power
        #modify steering_scale lower than 1.0 to have less responsive steering
        from donkeycar.parts.controller import get_js_controller

        ctr = get_js_controller(cfg)

        if cfg.USE_NETWORKED_JS:
            from donkeycar.parts.controller import JoyStickSub
            netwkJs = JoyStickSub(cfg.NETWORK_JS_SERVER_IP)
            V.add(netwkJs, threaded=True)
            ctr.js = netwkJs

    else:
        #This web controller will create a web server that is capable
        #of managing steering, throttle, and modes, and more.
        ctr = LocalWebController()


    V.add(ctr,
          inputs=['cam/image_array'],
          outputs=['user/angle', 'user/throttle', 'user/mode', 'recording', 'switch_mod', 'taga', 'tagb', 'tagc', 'tagd'],        #2020.10.24 by zmx
          threaded=True)

    #this throttle filter will allow one tap back for esc reverse
    th_filter = ThrottleFilter()
    V.add(th_filter, inputs=['user/throttle'], outputs=['user/throttle'])

    #See if we should even run the pilot module.
    #This is only needed because the part run_condition only accepts boolean
    class PilotCondition:
        def run(self, mode):
            if mode == 'user':
                return False
            else:
                return True

    V.add(PilotCondition(), inputs=['user/mode'], outputs=['run_pilot'])

    class LedConditionLogic:
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self, mode, recording, recording_alert, behavior_state, model_file_changed, track_loc):
            #returns a blink rate. 0 for off. -1 for on. positive for rate.

            if track_loc is not None:
                led.set_rgb(*self.cfg.LOC_COLORS[track_loc])
                return -1

            if model_file_changed:
                led.set_rgb(self.cfg.MODEL_RELOADED_LED_R, self.cfg.MODEL_RELOADED_LED_G, self.cfg.MODEL_RELOADED_LED_B)
                return 0.1
            else:
                led.set_rgb(self.cfg.LED_R, self.cfg.LED_G, self.cfg.LED_B)

            if recording_alert:
                led.set_rgb(*recording_alert)
                return self.cfg.REC_COUNT_ALERT_BLINK_RATE
            else:
                led.set_rgb(self.cfg.LED_R, self.cfg.LED_G, self.cfg.LED_B)

            if behavior_state is not None and model_type == 'behavior':
                r, g, b = self.cfg.BEHAVIOR_LED_COLORS[behavior_state]
                led.set_rgb(r, g, b)
                return -1 #solid on

            if recording:
                return -1 #solid on
            elif mode == 'user':
                return 1
            elif mode == 'local_angle':
                return 0.5
            elif mode == 'local':
                return 0.1
            return 0

    if cfg.HAVE_RGB_LED and not cfg.DONKEY_GYM:
        from donkeycar.parts.led_status import RGB_LED
        led = RGB_LED(cfg.LED_PIN_R, cfg.LED_PIN_G, cfg.LED_PIN_B, cfg.LED_INVERT)
        led.set_rgb(cfg.LED_R, cfg.LED_G, cfg.LED_B)

        V.add(LedConditionLogic(cfg), inputs=['user/mode', 'recording', "records/alert", 'behavior/state', 'modelfile/modified', "pilot/loc"],
              outputs=['led/blink_rate'])

        V.add(led, inputs=['led/blink_rate'])


    def get_record_alert_color(num_records):
        col = (0, 0, 0)
        for count, color in cfg.RECORD_ALERT_COLOR_ARR:
            if num_records >= count:
                col = color
        return col

    class RecordTracker:
        def __init__(self):
            self.last_num_rec_print = 0
            self.dur_alert = 0
            self.force_alert = 0

        def run(self, num_records):
            if num_records is None:
                return 0

            if self.last_num_rec_print != num_records or self.force_alert:
                self.last_num_rec_print = num_records

                if num_records % 10 == 0:
                    print("recorded", num_records, "records")

                if num_records % cfg.REC_COUNT_ALERT == 0 or self.force_alert:
                    self.dur_alert = num_records // cfg.REC_COUNT_ALERT * cfg.REC_COUNT_ALERT_CYC
                    self.force_alert = 0

            if self.dur_alert > 0:
                self.dur_alert -= 1

            if self.dur_alert != 0:
                return get_record_alert_color(num_records)

            return 0

    rec_tracker_part = RecordTracker()
    V.add(rec_tracker_part, inputs=["tub/num_records"], outputs=['records/alert'])

    if cfg.AUTO_RECORD_ON_THROTTLE and isinstance(ctr, JoystickController):
        #then we are not using the circle button. hijack that to force a record count indication
        def show_record_acount_status():
            rec_tracker_part.last_num_rec_print = 0
            rec_tracker_part.force_alert = 1
        ctr.set_button_down_trigger('circle', show_record_acount_status)

    #Sombrero
    if cfg.HAVE_SOMBRERO:
        from donkeycar.parts.sombrero import Sombrero
        s = Sombrero()

    #IMU
    if cfg.HAVE_IMU:
        from donkeycar.parts.imu import Mpu6050
        imu = Mpu6050()
        V.add(imu, outputs=['imu/acl_x', 'imu/acl_y', 'imu/acl_z',
            'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z'], threaded=True)

    #Behavioral state
    if cfg.TRAIN_BEHAVIORS:
        bh = BehaviorPart(cfg.BEHAVIOR_LIST)
        V.add(bh, outputs=['behavior/state', 'behavior/label', "behavior/one_hot_state_array"])
        try:
            ctr.set_button_down_trigger('L1', bh.increment_state)
        except:
            pass

        inputs = ['cam/image_array', "behavior/one_hot_state_array"]
    #IMU
    elif model_type == "imu":
        assert(cfg.HAVE_IMU)
        #Run the pilot if the mode is not user.
        inputs=['cam/image_array',
            'imu/acl_x', 'imu/acl_y', 'imu/acl_z',
            'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z']
    else:
        inputs=['cam/image_array']

    def load_model(kl, model_path):
        start = time.time()
        try:
            print('loading model', model_path)
            kl.load(model_path)
            print('finished loading in %s sec.' % (str(time.time() - start)) )
        except Exception as e:
            print(e)
            print('ERR>> problems loading model', model_path)

    def load_weights(kl, weights_path):
        start = time.time()
        try:
            print('loading model weights', weights_path)
            kl.model.load_weights(weights_path)
            print('finished loading in %s sec.' % (str(time.time() - start)) )
        except Exception as e:
            print(e)
            print('ERR>> problems loading weights', weights_path)

    def load_model_json(kl, json_fnm):
        start = time.time()
        print('loading model json', json_fnm)
        from tensorflow.python import keras
        try:
            with open(json_fnm, 'r') as handle:
                contents = handle.read()
                kl.model = keras.models.model_from_json(contents)
            print('finished loading json in %s sec.' % (str(time.time() - start)) )
        except Exception as e:
            print(e)
            print("ERR>> problems loading model json", json_fnm)

    if model_path:
        #When we have a model, first create an appropriate Keras part
        kl = dk.utils.get_model_by_type(model_type, cfg)

        model_reload_cb = None

        if '.h5' in model_path:
            #when we have a .h5 extension
            #load everything from the model file
            load_model(kl, model_path)

            def reload_model(filename):
                load_model(kl, filename)

            model_reload_cb = reload_model

        elif '.json' in model_path:
            #when we have a .json extension
            #load the model from there and look for a matching
            #.wts file with just weights
            load_model_json(kl, model_path)
            weights_path = model_path.replace('.json', '.weights')
            load_weights(kl, weights_path)

            def reload_weights(filename):
                weights_path = filename.replace('.json', '.weights')
                load_weights(kl, weights_path)

            model_reload_cb = reload_weights

        else:
            print("ERR>> Unknown extension type on model file!!")
            return
        
        
        #by zmx 2020.11.30 试图添加第二个模型
        
    if model_path_1:
        #When we have a model, first create an appropriate Keras part
        kl_1 = dk.utils.get_model_by_type(model_type, cfg)

        model_reload_cb_1 = None

        if '.h5' in model_path_1:
            #when we have a .h5 extension
            #load everything from the model file
            load_model(kl_1, model_path_1)

            def reload_model(filename):
                load_model(kl_1, filename)

            model_reload_cb_1 = reload_model

        elif '.json' in model_path_1:
            #when we have a .json extension
            #load the model from there and look for a matching
            #.wts file with just weights
            load_model_json(kl_1, model_path_1)
            weights_path_1 = model_path.replace('.json', '.weights')
            load_weights(kl_1, weights_path_1)

            def reload_weights(filename):
                weights_path_1 = filename.replace('.json', '.weights')
                load_weights(kl_1, weights_path_1)

            model_reload_cb_1 = reload_weights

        else:
            print("ERR>> Unknown extension type on model file!!")
            return

        #this part will signal visual LED, if connected
        V.add(FileWatcher(model_path, verbose=True), outputs=['modelfile/modified'])

        #these parts will reload the model file, but only when ai is running so we don't interrupt user driving
        V.add(FileWatcher(model_path), outputs=['modelfile/dirty'], run_condition="ai_running")
        V.add(DelayedTrigger(100), inputs=['modelfile/dirty'], outputs=['modelfile/reload'], run_condition="ai_running")
        V.add(TriggeredCallback(model_path, model_reload_cb), inputs=["modelfile/reload"], run_condition="ai_running")

        outputs_0=['pilot/angle_0', 'pilot/throttle_0']       #by zmx 2020.11.30 
        outputs_1=['pilot/angle_1', 'pilot/throttle_1']       #by zmx 2020.11.30 
        

        if cfg.TRAIN_LOCALIZER:
            outputs_0.append("pilot/loc")
            outputs_1.append("pilot/loc")

        V.add(kl, inputs=inputs,
            outputs=outputs_0,
            run_condition='run_pilot')
        
        V.add(kl_1, inputs=inputs,                  #by zmx 2020.11.30 把模型的的输出更名为 _1 和 _0 在选择器(MyPilot)里选择
            outputs=outputs_1,
            run_condition='run_pilot')

    class MySwitch:
        
        def __init__(self):
            self.pilot_state = 0  #by zmx 2020.10.24 到时候会改成int型
            self.mode = 0
            self.mode_switch = False

        def run(self,mod_switch):
            if (mod_switch):
                #print("detected")
                self.pilot_state = 1
            else:
                #print("not pushed")
                self.pilot_state = 0
            print("pushed : ",mod_switch,"\tmode : ",self.pilot_state)
            return self.pilot_state
        
        def update(self):
           while(True):
               self.pilot_state = self.run(self.mod_switch)
               
    

    #V.add(MySwitch(), inputs=['switch_mod'], outputs=['web_switch'])
    
    class MyButton:
        
        def __init__(self):
            self.button_in = [False,False,False,False]
            self.state_out = [True,False,False,False]      #by zmx 2020.12.7添加了4个状态量bool，其中默认0是默认状态
            self.state_out1 = 1                          #by zmx 2020.12.13 默认是n，接到后变化
        def run(self,a,b,c,d):
            if a:
                self.state_out1 = 1
                self.state_out = [True,False,False,False]
                print("A triggered")
            elif b:
                self.state_out1 = 2
                self.state_out = [False,True,False,False]
                print("B triggered")
            elif c:
                self.state_out1 = 3
                self.state_out = [False,False,True,False]
                print("X triggered")
            elif d:
                self.state_out1 = 4
                self.state_out = [False,False,False,True]
                print("Y triggered")
            #print(self.state_out1)
            return self.state_out1
        def update(self):
            while(True):
                a = self.button_in[0]
                b = self.button_in[1]
                c = self.button_in[2]
                d = self.button_in[3]
                self.state_out1 = self.run(a,b,c,d)
        def run_threaded(self,taga,tagb,tagc,tagd):
            self.button_in = [taga,tagb,tagc,tagd]
            #print(self.state_out1,"---")
            return self.state_out1
        
        
    V.add(MyButton(),inputs=['taga','tagb','tagc','tagd'],outputs=['my_state'],run_condition='switch_mod')
            
    class MyPilot:
        def __init__(self):
            self.angle = 0  
            self.throttle = 0
            self.angle_0 = 0
            self.angle_1 = 0
            self.throttle_0 = 0
            self.throttle_1 = 0
            self.mode = 0
            self.state = [False,False,False,False]
            self.state1 = 0
            

        def run_threaded(self,my_state,a0,t0,a1,t1):
            self.angle_0 = a0
            self.angle_1 = a1
            self.throttle_0 = t0
            self.throttle_1 = t1
            #self.mode = mode
            self.state1 = my_state
            return self.angle, self.throttle

        def update(self):
            while True:
                self.angle, self.throttle = self.run(self.state1,self.angle_0,self.angle_1,self.throttle_0,self.throttle_1)
                #print(self.angle)

        def run(self,state1,angle_0,throttle_0,angle_1,throttle_1):
            print("mystate:",state1,"mode11::",angle_0,throttle_0,"model2:",angle_1,throttle_1)
            if state1==1:
                #print("mode:",pilot_mode," angle:",angle_0," throttle:",throttle_0)
                return(angle_0,throttle_0)
            elif state1==2:
                #print("mode:",pilot_mode," angle:",angle_1," throttle:",throttle_1)
                return(angle_1,throttle_1)
            else:
                return(0,0)

    V.add(MyPilot(), inputs=['my_state','pilot/angle_0', 'pilot/throttle_0','pilot/angle_1', 'pilot/throttle_1'], outputs=['mypilot/angle','mypilot/throttle'],run_condition='switch_mod')

    #Choose what inputs should change the car.
    class DriveMode:
        def run(self, mode,
                    user_angle, user_throttle,
                    mypilot_angle, mypilot_throttle):
            #print("drivemode is running")
            #print(taga,"!!!!!!!!",pilot_mode)
            if mode == 'user':
                #print(taga,"!!!!!!!!",pilot_mode)
                
                return user_angle, user_throttle
                

            elif mode == 'local_angle':
                #print("local_angle")
                return mypilot_angle, user_throttle

            else:
                
                #print("mode:",pilot_mode," angle:",mypilot_angle," throttle:",mypilot_throttle)
                return mypilot_angle, mypilot_throttle
                #return mypilot_angle, mypilot_throttle * cfg.AI_THROTTLE_MULT      #by zmx 2020.11.30 临时的修改，把my的输出强加上去看看

    V.add(DriveMode(),
          inputs=['user/mode', 'user/angle', 'user/throttle',
                  'mypilot/angle', 'mypilot/throttle'],
          outputs=['angle', 'throttle'])


    #to give the car a boost when starting ai mode in a race.
    aiLauncher = AiLaunch(cfg.AI_LAUNCH_DURATION, cfg.AI_LAUNCH_THROTTLE, cfg.AI_LAUNCH_KEEP_ENABLED)

    V.add(aiLauncher,
        inputs=['user/mode', 'throttle'],
        outputs=['throttle'])

    if isinstance(ctr, JoystickController):
        ctr.set_button_down_trigger(cfg.AI_LAUNCH_ENABLE_BUTTON, aiLauncher.enable_ai_launch)


    class AiRunCondition:
        '''
        A bool part to let us know when ai is running.
        '''
        def run(self, mode):
            if mode == "user":
                return False
            return True

    V.add(AiRunCondition(), inputs=['user/mode'], outputs=['ai_running'])

    #Ai Recording
    class AiRecordingCondition:
        '''
        return True when ai mode, otherwize respect user mode recording flag
        '''
        def run(self, mode, recording):
            if mode == 'user':
                return recording
            return True

    if cfg.RECORD_DURING_AI:
        V.add(AiRecordingCondition(), inputs=['user/mode', 'recording'], outputs=['recording'])

    #Drive train setup
    if cfg.DONKEY_GYM:
        pass

    elif cfg.DRIVE_TRAIN_TYPE == "SERVO_ESC":
        from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle

        steering_controller = PCA9685(cfg.STEERING_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
        steering = PWMSteering(controller=steering_controller,
                                        left_pulse=cfg.STEERING_LEFT_PWM,
                                        right_pulse=cfg.STEERING_RIGHT_PWM)

        throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
        throttle = PWMThrottle(controller=throttle_controller,
                                        max_pulse=cfg.THROTTLE_FORWARD_PWM,
                                        zero_pulse=cfg.THROTTLE_STOPPED_PWM,
                                        min_pulse=cfg.THROTTLE_REVERSE_PWM)

        V.add(steering, inputs=['angle'])
        V.add(throttle, inputs=['throttle'])


    elif cfg.DRIVE_TRAIN_TYPE == "DC_STEER_THROTTLE":
        from donkeycar.parts.actuator import Mini_HBridge_DC_Motor_PWM

        steering = Mini_HBridge_DC_Motor_PWM(cfg.HBRIDGE_PIN_LEFT, cfg.HBRIDGE_PIN_RIGHT)
        throttle = Mini_HBridge_DC_Motor_PWM(cfg.HBRIDGE_PIN_FWD, cfg.HBRIDGE_PIN_BWD)

        V.add(steering, inputs=['angle'])
        V.add(throttle, inputs=['throttle'])


    elif cfg.DRIVE_TRAIN_TYPE == "DC_TWO_WHEEL":
        from donkeycar.parts.actuator import TwoWheelSteeringThrottle, Mini_HBridge_DC_Motor_PWM

        left_motor = Mini_HBridge_DC_Motor_PWM(cfg.HBRIDGE_PIN_LEFT_FWD, cfg.HBRIDGE_PIN_LEFT_BWD)
        right_motor = Mini_HBridge_DC_Motor_PWM(cfg.HBRIDGE_PIN_RIGHT_FWD, cfg.HBRIDGE_PIN_RIGHT_BWD)
        two_wheel_control = TwoWheelSteeringThrottle()

        V.add(two_wheel_control,
                inputs=['throttle', 'angle'],
                outputs=['left_motor_speed', 'right_motor_speed'])

        V.add(left_motor, inputs=['left_motor_speed'])
        V.add(right_motor, inputs=['right_motor_speed'])

    elif cfg.DRIVE_TRAIN_TYPE == "SERVO_HBRIDGE_PWM":
        from donkeycar.parts.actuator import ServoBlaster, PWMSteering
        steering_controller = ServoBlaster(cfg.STEERING_CHANNEL) #really pin
        #PWM pulse values should be in the range of 100 to 200
        assert(cfg.STEERING_LEFT_PWM <= 200)
        assert(cfg.STEERING_RIGHT_PWM <= 200)
        steering = PWMSteering(controller=steering_controller,
                                        left_pulse=cfg.STEERING_LEFT_PWM,
                                        right_pulse=cfg.STEERING_RIGHT_PWM)


        from donkeycar.parts.actuator import Mini_HBridge_DC_Motor_PWM
        motor = Mini_HBridge_DC_Motor_PWM(cfg.HBRIDGE_PIN_FWD, cfg.HBRIDGE_PIN_BWD)

        V.add(steering, inputs=['angle'])
        V.add(motor, inputs=["throttle"])


    #add tub to save data

    inputs=['cam/image_array',
            'user/angle', 'user/throttle',
            'user/mode'
            ]

    types=['image_array',
           'float', 'float',
           'str'
           ]

    if cfg.TRAIN_BEHAVIORS:
        inputs += ['behavior/state', 'behavior/label', "behavior/one_hot_state_array"]
        types += ['int', 'str', 'vector']
        
    if cfg.HAVE_GAMEPAD:
        inputs += ['my_state']
        types += ['int']

    if cfg.HAVE_IMU:
        inputs += ['imu/acl_x', 'imu/acl_y', 'imu/acl_z',
            'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z']

        types +=['float', 'float', 'float',
           'float', 'float', 'float']

    if cfg.RECORD_DURING_AI:
        inputs += ['pilot/angle', 'pilot/throttle']
        types += ['float', 'float']

    th = TubHandler(path=cfg.DATA_PATH)
    tub = th.new_tub_writer(inputs=inputs, types=types, user_meta=meta)
    V.add(tub, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')

    if cfg.PUB_CAMERA_IMAGES:
        from donkeycar.parts.network import TCPServeValue
        from donkeycar.parts.image import ImgArrToJpg
        pub = TCPServeValue("camera")
        V.add(ImgArrToJpg(), inputs=['cam/image_array'], outputs=['jpg/bin'])
        V.add(pub, inputs=['jpg/bin'])

    if type(ctr) is LocalWebController:
        print("You can now go to <your pi ip address>:8887 to drive your car.")
    elif isinstance(ctr, JoystickController):
        print("You can now move your joystick to drive your car.")
        #tell the controller about the tub
        ctr.set_tub(tub)

        if cfg.BUTTON_PRESS_NEW_TUB:

            def new_tub_dir():
                V.parts.pop()
                tub = th.new_tub_writer(inputs=inputs, types=types, user_meta=meta)
                V.add(tub, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')
                ctr.set_tub(tub)

            ctr.set_button_down_trigger('cross', new_tub_dir)
        ctr.print_controls()

    #run the vehicle for 20 seconds
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ,
            max_loop_count=cfg.MAX_LOOPS)


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config()

    if args['drive']:
        model_type = args['--type']
        camera_type = args['--camera']
        drive(cfg, 
              model_path=args['--model'],
              model_path_1=args['--model_1'],        #by zmx 2020.11.29 试图添加第二个模型 
              use_joystick=args['--js'],
              model_type=model_type,
              camera_type=camera_type,
              meta=args['--meta'])

    if args['train']:
        from train import multi_train, preprocessFileList

        tub = args['--tub']
        model = args['--model']
        transfer = args['--transfer']
        model_type = args['--type']
        continuous = args['--continuous']
        aug = args['--aug']

        dirs = preprocessFileList( args['--file'] )
        if tub is not None:
            tub_paths = [os.path.expanduser(n) for n in tub.split(',')]
            dirs.extend( tub_paths )

        multi_train(cfg, dirs, model, transfer, model_type, continuous, aug)

