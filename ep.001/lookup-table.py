# A system that satisfies every requirement it was given...
# by memorizing the exact answer. No understanding, no generalization.

requirements_to_solutions = {
    "2+2": 4,
    "3+5": 8,
    "10-4": 6,
}

def lookup_solver(requirement: str):
    if requirement in requirements_to_solutions:
        return requirements_to_solutions[requirement]
    return None  # anything it wasn't explicitly given, it cannot solve

print(lookup_solver("2+2"))   # 4  -- looks "intelligent"
print(lookup_solver("7+1"))   # None -- never seen this exact requirement, total failure