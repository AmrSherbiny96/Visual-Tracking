import numpy as np
import cv2

class ParticleFilter():
    def __init__(self, n_particles, control_sigma, mse_sigma, image_shape, 
                template, dynamic_appearance_model=False, alpha=0):
        self.n_particles = n_particles
        self.control_sigma = control_sigma
        self.mse_sigma = mse_sigma
        self.state = np.zeros(2, dtype=np.float)
        self.template = template

        frame_width, frame_height = image_shape[0], image_shape[1]
        
        #calculate the min and max height to ensure particles don't estimate patches outside the frame
        self.min_height, self.max_height = template.shape[0]//2, frame_width - template.shape[0]//2
        self.min_width, self.max_width = template.shape[1]//2, frame_height - template.shape[1]//2
        
        self.dynamic_appearance_model = dynamic_appearance_model
        self.alpha = alpha

        self.init_particles()

    def init_particles(self):
        '''Initialize particles with at random positions 
            with uniform weights'''
        random_x = np.random.uniform(self.min_width, self.max_width, self.n_particles)
        random_y = np.random.uniform(self.min_height, self.max_height, self.n_particles)
        self.particles = np.array([(i,j) for i,j in zip(random_x, random_y)])
        self.weights = np.array([1/len(self.particles) for i in range(self.n_particles)])

    def move(self):
        '''Move particles by adding random values
            sampled from normal distribution with 
            standard deviation = control_sigma'''
        self.particles[:,0] += np.random.normal(scale=self.control_sigma, size=self.n_particles)
        self.particles[:,1] += np.random.normal(scale=self.control_sigma, size=self.n_particles)

    def calc_measurement_likelihood(self, patch):
        if(patch.shape != self.template.shape):
            return 0

        mse = np.sum(np.subtract(self.template, patch, dtype=np.float32) ** 2)
        mse /= float(self.template.shape[0] * self.template.shape[1])

        return np.exp(-mse/(2 * np.square(self.mse_sigma)))

    def measure(self, frame):
        '''Get measurment of each particle. Then
            compute measurement likelihood and 
            update weights accordingly.'''
        y_boundary, x_boundary = self.template.shape
        
        for i, particle in enumerate(self.particles):
            y_min = (particle[1]-y_boundary//2).astype(np.int) 
            x_min = (particle[0]-x_boundary//2).astype(np.int)
            patch = frame[y_min: y_min+y_boundary , x_min:x_min+x_boundary]
            self.weights[i] = self.calc_measurement_likelihood(patch)
            
        self.normalize_weights()


    def resample(self):
        '''Resample particles with updated weights.
            Particles with higher measurement likelihoods
            have higher weights and therefore are more
            likely to survive.'''
        resampled_indices = np.random.choice(self.particles.shape[0], size=self.n_particles, replace=True, p=self.weights)
        self.particles = self.particles[resampled_indices, :]
        self.weights = self.weights[resampled_indices]
        self.normalize_weights()

    def normalize_weights(self):
        '''Normalize weights after weight updates'''
        self.weights = self.weights / np.sum(self.weights)

    def estimate_state(self):

        tmp = self.particles.copy()
        tmp[:,0] = np.multiply(tmp[:,0], self.weights)
        tmp[:,1] = np.multiply(tmp[:,1], self.weights)
        self.state = np.sum(tmp, axis=0)

    def update_model(self, frame):
        '''Update appearance model to be
            a combination of the previous
            template and the current predicted 
            location.'''
        y_est, x_est = self.state[1], self.state[0]
        y_max, x_max = self.template.shape[:2]

        y_min = (y_est - y_max//2).astype(np.int) 
        x_min = (x_est - x_max//2).astype(np.int)
        estimated_template = frame[y_min:y_min+y_max, x_min:x_min+x_max]

        tmp = self.template.copy()
        self.template = (self.alpha*estimated_template + (1-self.alpha)*self.template).astype(np.int32)
    
    def update(self, frame):
        '''Perform particle filter updates'''
        self.move()
        self.measure(frame)
        self.resample()
        self.estimate_state()
        
        if(self.dynamic_appearance_model):
            self.update_model(frame)

        self.visualize_particles(frame)

    def visualize_particles(self, frame):
        '''Visualize particle locations
            on the frame'''
        estimated_x = self.state[0]
        estimated_y = self.state[1]

        y_boundary, x_boundary = self.template.shape
        y_min = (estimated_y -y_boundary//2).astype(np.int) 
        x_min = (estimated_x-x_boundary//2).astype(np.int)

        cv2.rectangle(frame, (x_min, y_min), (x_min+x_boundary, y_min+y_boundary), (255,0,0), 1)

        for i, particle in enumerate(self.particles):
            x = int(particle[0])
            y = int(particle[1])
            cv2.circle(frame, (x,y), 1, (255,0,0))

        return frame
	

    