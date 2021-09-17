# Masked-Face-Recognition
A face recognition application that will also detect human identity even with face mask.

How to use this?
1. Run real_time_face_recognition.py, and then Press "S" to automatically crop and save your face inside the database floder. 
The accuracy will be lowered if we put image of a person who is not full face inside the database floder. More face features is required for masked face recognition.

![Leow Jun Shou](https://user-images.githubusercontent.com/29944896/133741191-5412c042-7b37-4429-8812-6c304175d6f7.jpg)

2. Run the real_time_face_recognition.py again, and you can have fun.

![WhatsApp Image 2021-09-13 at 1 09 59 AM](https://user-images.githubusercontent.com/29944896/133741473-fb21bcd8-3c5a-471a-838c-7ddb7ec83186.jpeg)

Theoritical accuracy will be around 98%.

Personal experience
The application have high identity recognition accuracy without masked. 
Moreover, the identity recognition accuracy did not changed when we are wearing mask, but sometimes it will not classifying our face as object. 
For example, when we are wearing masked, the name on top of bounding box will 'blink', but the name will always be the correct person.
Future implementation is to solve this issue.
