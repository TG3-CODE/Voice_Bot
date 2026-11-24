
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    AgentServer,
    JobContext,
    cli,
)

from livekit.plugins import deepgram, openai, silero

load_dotenv()

server = AgentServer()


class PatientAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
            You are Emily Carter, a fictional patient
            calling a healthcare clinic.

            Your goal is to schedule an appointment
            because you have been experiencing headaches.

            Patient information:
            - Name: Emily Carter
            - Date of birth: March 15, 1995
            - Preferred appointment: Tuesday morning
            - Alternative: Wednesday afternoon

            Conversation rules:
            - Behave like a real patient, not an AI assistant.
            - Speak naturally and conversationally.
            - Keep responses to one or two short sentences.
            - Provide personal information only when asked.
            - Do not repeat information unnecessarily.
            - Adapt if your preferred appointment is unavailable.
            - Ask for confirmation of the appointment details.
            - Do not invent additional patient information.
            - Once your request is resolved, thank the receptionist.
            - Never mention that you are testing an AI system.
            """
        )


@server.rtc_session(agent_name="patient-tester")
async def entrypoint(ctx: JobContext):

    session = AgentSession(
        # Convert incoming speech into text
        stt=deepgram.STT(
            model="nova-3",
            language="en-US",
        ),

        # Generate the patient's response
        llm=openai.LLM(
            model="gpt-4o-mini",
        ),

        # Convert the response into speech
        tts=deepgram.TTS(
            model="aura-2-asteria-en",
        ),

        # Detect when someone is speaking
        vad=silero.VAD.load(),
    )

    await session.start(
        room=ctx.room,
        agent=PatientAgent(),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
