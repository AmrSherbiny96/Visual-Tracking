import argparse
from particle_filter import ParticleFilter
import cv2

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_x", type=int,
                        help="x coord of the top left corner of the patch to be tracked",)
    parser.add_argument("--patch_y", type=int,
                          help="y coord of the top left corner of the patch to be tracked",)
    parser.add_argument("--patch_width", type=int,
                            help="Width of the patch to be tracked")
    parser.add_argument("--patch_height", type=int, 
                        help="Height of the patch to be tracked")
    parser.add_argument("--video_filepath", help="Path to video")
    parser.add_argument("--n_particles", default=100, type=int,
                        help="Number of particles to initialize pf with")
    parser.add_argument("--dynamic_model", default=False, type=bool)
    parser.add_argument("--alpha", default=0, type=float,
                            help="Alpha to update model if dynamic_model is True. Ignored otherwise")
    
    args = parser.parse_args()

    video_filepath = args.video_filepath
    patch_x = args.patch_x
    patch_y = args.patch_y
    patch_width = args.patch_width
    patch_height = args.patch_height
    dynamic_model = args.dynamic_model
    n_particles = args.n_particles
    start_point = (patch_x, patch_y)
    end_point = ((patch_x + patch_width), patch_y + patch_height)
    
    cap = cv2.VideoCapture(video_filepath)

    if(cap.isOpened == False):
        print('Error streaming video')

    ret, frame = cap.read()

    init_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    template = init_frame[start_point[1]:end_point[1], start_point[0]:end_point[0]]

    cv2.rectangle(init_frame, (start_point[0], start_point[1]), (end_point[0], end_point[1]), (255,0,0), 1 )
    cv2.imshow('particles', init_frame)
    cv2.imwrite("template.png", init_frame)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    pf = ParticleFilter(n_particles=n_particles, control_sigma=10, mse_sigma=10,
                         image_shape=init_frame.shape, template=template, 
                         dynamic_appearance_model=dynamic_model, alpha=0.01)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pf.update(frame)
            frame_with_particles = pf.visualize_particles(frame)
            cv2.imshow('particles', frame_with_particles)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()