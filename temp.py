from environment import GripperEnv


env = GripperEnv()

print("Checking reset logic:")
print(f"{'Target':>8} | {'Stiffness':>10} | {'Contact':>8} | {'Angle4Target':>13} | {'StartAngle':>11} | {'StartForce':>11}")
print("-" * 75)

for _ in range(10):
    env.reset(target_force=3.0)
    angle_for_target = env.contact_angle + (env.target_force / env.stiffness)
    f1, f2 = env._simulate_forces()
    avg = (f1 + f2) / 2.0
    print(f"{env.target_force:>8.2f} | {env.stiffness:>10.4f} | {env.contact_angle:>8.1f} | "
          f"{angle_for_target:>13.1f} | {env.angle:>11.1f} | {avg:>11.3f}N")