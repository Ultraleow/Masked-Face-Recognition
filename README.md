# Masked-Face-Recognition
A face recognition application that will also detect human identity even with face mask.

How to use this?
1. Run real_time_face_recognition.py, and then Press "S" to automatically crop and save your face inside the database floder. 
The accuracy will be lowered if we put image of a person who is not full face inside the database floder. More face features is required for masked face recognition.

2. Run the real_time_face_recognition.py again, and you can have fun.

Personal experience
The application have high identity recognition accuracy without masked. 
Moreover, the identity recognition accuracy did not changed when we are wearing mask, but sometimes it will not classifying our face as object. 
For example, when we are wearing masked, the name on top of bounding box will 'blink', but the name will always be the correct person.
Future implementation is to solve this issue.
