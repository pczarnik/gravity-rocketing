import math

import numpy as np

import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils.step_api_compatibility import step_api_compatibility

import Box2D
from Box2D.b2 import (
    circleShape,
    contactListener,
    edgeShape,
    fixtureDef,
    polygonShape,
    vec2,
    revoluteJointDef,
)

FPS = 30
VIEWPORT_W = 600
VIEWPORT_H = 400
VIEWPORT_MIN = min(VIEWPORT_W, VIEWPORT_H)
TRANSFORM_VEC = lambda xy: vec2(
    (VIEWPORT_W - VIEWPORT_MIN) / 2 + VIEWPORT_MIN / 2 * (xy[0] + 1),
    (VIEWPORT_H - VIEWPORT_MIN) / 2 + VIEWPORT_MIN / 2 * (xy[1] + 1),
)
TRANSFORM_RADIUS = lambda r: r * VIEWPORT_MIN / 2
GRAVITY_CONST = 1e-3
PLANET_DENS = 10


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if (
            self.env.rocket == contact.fixtureA.body
            or self.env.rocket == contact.fixtureB.body
        ):
            print("Rocket hit the planet!")
            self.env.game_over = True

    def EndContact(self, contact):
        pass


class Rocket(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(self, rocket_pos_ang_size_mass, planets_pos_mass, render_mode=None):
        self.rocket_pos_ang_size_mass = rocket_pos_ang_size_mass
        self.planets_pos_mass = planets_pos_mass
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.world = None

        low = np.array(
            [
                -10.0,  # v_x
                -10.0,  # v_y
                -2 * math.pi,  # theta
                -10.0,  # theta_dot
            ] + [0] * len(self.planets_pos_mass)  # 0 is the min distance
        ).astype(np.float32)

        high = np.array(
            [
                10.0,  # v_x
                10.0,  # v_y
                2 * math.pi,  # theta
                10.0,  # theta_dot
            ] + [4] * len(self.planets_pos_mass)  # 4 is the max distance
        ).astype(np.float32)

        self.observation_space = gym.spaces.Box(low, high)
        self.action_space = gym.spaces.Discrete(4)

    def _destroy(self):
        if self.world is None:
            return
        self.world.contactListener = None
        self.world.DestroyBody(self.rocket)
        self.rocket = None
        for planet in self.planets:
            self.world.DestroyBody(planet)
        self.planets = []
        self.world = None

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self._destroy()

        self.world = Box2D.b2World(gravity=(0, 0))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False

        rocket_pos = self.rocket_pos_ang_size_mass[:2]
        rocket_deg = self.rocket_pos_ang_size_mass[2]
        rocket_size = self.rocket_pos_ang_size_mass[3]
        self.rocket_mass = self.rocket_pos_ang_size_mass[4]
        rocket_shape = [
            (rocket_size, 0),
            (-rocket_size, 0),
            (0, 2.5*rocket_size)
        ]
        self.rocket = self.world.CreateDynamicBody(
            position=rocket_pos,
            angle=rocket_deg,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=rocket_shape),
                density=self.rocket_mass,
            )
        )
        self.rocket.color1 = (128, 128, 128)
        self.rocket.color2 = (128, 128, 128)

        self.planets = []
        for planet_pos_mass in self.planets_pos_mass:
            planet = self.world.CreateStaticBody(
                position=planet_pos_mass[:2], angle=0.0,
                fixtures=fixtureDef(
                    shape=circleShape(radius=planet_pos_mass[2] / PLANET_DENS)
                )
            )
            planet.color1 = (255, 255, 255)
            planet.color2 = (255, 255, 255)
            self.planets.append(planet)

        self.drawlist = [self.rocket] + self.planets

        if self.render_mode == "human":
            self.render()

        # self.rocket.ApplyTorque(
        #     0.000005,
        #     True,
        # )

        self.rocket.linearVelocity = vec2(0.5, 0)

        return self.step(0)[0], {}

    def _get_observation(self):
        rocket_pos = self.rocket.position
        rocket_v = self.rocket.linearVelocity
        rocket_theta = self.rocket.angle
        rocket_theta_dot = self.rocket.angularVelocity

        planets_pos = [planet.position for planet in self.planets]

        return np.array(
            [
                rocket_v.x,
                rocket_v.y,
                rocket_theta,
                rocket_theta_dot,
            ]
            + [np.linalg.norm(rocket_pos - planet_pos) for planet_pos in planets_pos]
        )

    def step(self, action):
        assert self.rocket is not None
        assert self.action_space.contains(action)

        obs = self._get_observation()

        acc = vec2(0, 0)

        for (x, y, mass) in self.planets_pos_mass:
            delta = vec2(x, y) - self.rocket.position
            dist = np.linalg.norm(delta)

            acc += delta * mass / (dist ** 3)

        acc *= GRAVITY_CONST * self.rocket_mass

        self.rocket.ApplyForceToCenter(acc, True)

        # self.rocket.ApplyLinearImpulse(
        #     0.000001 * self.rocket.linearVelocity / (np.linalg.norm(self.rocket.linearVelocity) + 1e-6),
        #     self.rocket.position,
        #     True
        # )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        reward = 0
        terminated = False
        if self.game_over:
            reward = -100
            terminated = True

        return obs, reward, terminated, False, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[box2d]"`'
            ) from e

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))

        self.surf.fill((0, 0, 0))

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=TRANSFORM_VEC(trans * f.shape.pos),
                        radius=TRANSFORM_RADIUS(f.shape.radius),
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color2,
                        center=TRANSFORM_VEC(trans * f.shape.pos),
                        radius=TRANSFORM_RADIUS(f.shape.radius),
                    )

                else:
                    path = [TRANSFORM_VEC(trans * v) for v in f.shape.vertices]
                    pygame.draw.polygon(self.surf, color=obj.color1, points=path)
                    gfxdraw.aapolygon(self.surf, path, obj.color1)
                    pygame.draw.aalines(
                        self.surf, color=obj.color2, points=path, closed=True
                    )

        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )


def demo_rocket(env, render=False):
    s, info = env.reset()
    steps = 0
    while True:
        s, r, terminated, truncated, info = step_api_compatibility(env.step(0), True)

        if render:
            still_open = env.render()
            if still_open is False:
                break

            steps += 1

        if terminated or truncated:
            break

    if render:
        env.close()


if __name__ == "__main__":
    env = Rocket(
        rocket_pos_ang_size_mass=(-0.75, -0.75, -np.pi/4, 0.03, 0.1),
        planets_pos_mass=[
            (-0.5, 0.5, 2),
            # (0.5, -0.5, 3),
            # (0.5, 0.5, 0.5)
        ],
        render_mode="human"
    )
    demo_rocket(env, render=True)
