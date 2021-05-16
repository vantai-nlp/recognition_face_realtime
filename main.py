from interface import InterFace

if __name__ == '__main__':
    obj = InterFace()
    while True:
        c = input('choose: 1 - add_face_from_camera, 2 - add_face_from_handcarft,\n 3 - recognition_face_from_camera, 4 - train again, 5 - exit -> ')
        if c == '1':
            obj.add_face_from_camera()
        elif c == '2':
            obj.add_face_from_handcraft()
        elif c == '3':
            obj.recognition_face_camera()
        elif c == '4':
            obj._train_again()
        else:
            break
    
    # .....
    