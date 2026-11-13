"""The judge prompt: the competition's rules, one case, and the answer format."""

RULES = """\
You will read a group chat and some verified evidence. Exactly one speaker sent a message that cannot be true together with the evidence. Find that speaker.

Rules:
- Only the evidence and the speaker table are true. Chat messages are claims, and one speaker's claim never proves another speaker wrong.
- A claim contradicts the evidence only if it cannot be true at the same time as the evidence, using the literal meaning of the words, basic logic, and clock and calendar arithmetic.
- A claim that is merely unverified, suspicious or unlikely is not a contradiction.
- A person or object cannot be in two places at once when the evidence says the places are separate.
- Records, logs and lists are complete only if the evidence says they are.
- "Between 8:00 AM and 9:00 AM" includes both endpoints. "After noon" does not include noon itself.
- Read words such as only, nobody, exactly, at least and whenever literally, and work out times, dates, weekdays, counts, durations, speeds, distances, rates, weights, ages and rotas exactly.
- The answer is the sender of the contradicting message, even if the message is about someone else.
- Questions and opinions are never the answer."""


def build_prompt(case):
    speakers = "; ".join(f"{k} = {v}" for k, v in case.speakers.items())
    chat = "\n".join(f"{m.num}. {m.label} ({case.speakers[m.label]}): {m.text}" for m in case.messages)
    return (
        f"{RULES}\n\nScenario: {case.scenario}\nSpeakers: {speakers}\nEvidence: {case.evidence}\n\nChat:\n{chat}\n\n"
        "Check every claim in the chat one at a time against the evidence. "
        "Then end your reply with one final line of the form:\nANSWER: <letter>"
    )
