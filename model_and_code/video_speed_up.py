from moviepy.editor import *
intp_name = 'C:/Users/XIR1SBY/Desktop/result/good_result_xie.mp4'
outp_name = 'C:/Users/XIR1SBY/Desktop/result/good_result_xie_speed_up.mp4'
play_speed = 15.0 #速率
au = VideoFileClip(intp_name)
new_au = au.fl_time(lambda t:  play_speed*t, apply_to=['mask', 'audio'])
new_au = new_au.set_duration(au.duration/play_speed)
new_au.write_videofile(outp_name)
