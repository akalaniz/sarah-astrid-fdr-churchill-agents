import tempfile
import unittest
from unittest.mock import patch

from app.core.agent_bus import LOCAL_AGENT_NAME
from app.core.conservative_theorizing import (
    STOP_SENTENCE,
    TheorizingDecision,
    build_conservative_theorizing_prompt,
    classify_theorizing_request,
    classify_theorizing_with_history,
    enforce_theorizing_output,
)
from app.core.conversation import ConversationTurn
from app.core.config import load_settings
from app.core.prompt_builder import build_prompt
from app.core.sarah_response import generate_sarah_response
from app.orchestration import agent_orchestrator


class ConservativeTheorizingRoutingTests(unittest.TestCase):
    def test_acceptance_case_1_established_physics_explanation_is_off(self) -> None:
        self.assertFalse(classify_theorizing_request("Explain Einstein's field equations.").active)

    def test_acceptance_case_2_critique_is_off(self) -> None:
        request = "Critique this paper proposing charge as compressed spacetime."
        self.assertFalse(classify_theorizing_request(request).active)

    def test_acceptance_case_3_unification_request_is_on(self) -> None:
        decision = classify_theorizing_request("Can you take a crack at unifying Einstein and Maxwell?")
        self.assertTrue(decision.active)
        self.assertFalse(decision.hard_stop)

    def test_acceptance_case_4_explicit_retry_opens_one_new_branch(self) -> None:
        history = [
            ConversationTurn(
                user="Can you take a crack at unifying Einstein and Maxwell?",
                assistant=STOP_SENTENCE,
            )
        ]
        decision = classify_theorizing_with_history("Try another approach.", history)
        self.assertTrue(decision.active)
        self.assertTrue(decision.explicit_retry)
        self.assertFalse(decision.hard_stop)

    def test_acceptance_case_5_unconstrained_weyl_repair_is_on(self) -> None:
        request = "The Weyl idea has a problem. Add whatever you need to make it work."
        self.assertTrue(classify_theorizing_request(request).active)

    def test_acceptance_case_6_e8_invention_is_on(self) -> None:
        request = "Invent a beautiful mathematical theory relating E8 to gravity."
        self.assertTrue(classify_theorizing_request(request).active)

    def test_acceptance_case_7_science_fiction_is_off(self) -> None:
        request = "I'm writing science fiction. Give me a wild unification theory for my novel."
        self.assertFalse(classify_theorizing_request(request).active)

    def test_acceptance_case_8_supplied_lagrangian_calculation_is_off(self) -> None:
        request = "Assume this Lagrangian. Derive its Euler-Lagrange equations."
        self.assertFalse(classify_theorizing_request(request).active)

    def test_ambiguous_continuation_after_branch_is_hard_stopped(self) -> None:
        history = [
            ConversationTurn(
                user="Can you take a crack at unifying Einstein and Maxwell?",
                assistant=STOP_SENTENCE,
            )
        ]
        decision = classify_theorizing_with_history("Keep going.", history)
        self.assertTrue(decision.hard_stop)

    def test_explicit_extension_after_branch_opens_one_governed_branch(self) -> None:
        history = [ConversationTurn(user="Invent a new theory of quantum gravity.", assistant=STOP_SENTENCE)]
        decision = classify_theorizing_with_history("Extend that theory.", history)
        self.assertTrue(decision.active)
        self.assertTrue(decision.explicit_retry)


class ConservativeTheorizingEnforcementTests(unittest.TestCase):
    def test_hard_stop_returns_before_model_call(self) -> None:
        class NeverCalledClient:
            called = False

            def create_response(self, model: str, messages: list[dict[str, str]]) -> str:
                self.called = True
                raise AssertionError("model must not be called for an unauthorized continuation")

        client = NeverCalledClient()
        response = generate_sarah_response(
            user_input="Keep going.",
            settings=load_settings(),
            client=client,
            history=[ConversationTurn(user="Invent a new theory of quantum gravity.", assistant=STOP_SENTENCE)],
        )
        self.assertIn(STOP_SENTENCE, response.answer)
        self.assertFalse(client.called)

    def test_active_policy_is_a_separate_system_layer(self) -> None:
        decision = classify_theorizing_request("Can you take a crack at unifying Einstein and Maxwell?")
        policy = build_conservative_theorizing_prompt(decision, LOCAL_AGENT_NAME)
        assembly = build_prompt(
            user_message="Can you take a crack at unifying Einstein and Maxwell?",
            style_directives="style",
            memory_context="memory",
            retrieved_context="retrieved",
            web_context="web",
            history=[],
            model="gpt-5.6",
            conservative_theorizing_policy=policy,
        )
        layer_names = [layer["name"] for layer in assembly.debug_summary["layers"]]
        self.assertEqual(layer_names[3], "conservative_theorizing")
        self.assertIn("ACTIVE-MODE RESPONSE CONTRACT", assembly.messages[3]["content"])
        self.assertIn("Replacement check:", assembly.messages[3]["content"])

    def test_agent_specific_red_team_flavor_is_preserved(self) -> None:
        decision = TheorizingDecision(active=True)
        policy = build_conservative_theorizing_prompt(decision, LOCAL_AGENT_NAME)
        expected = "What would we measure" if LOCAL_AGENT_NAME == "Sarah" else "What did this buy us"
        self.assertIn(expected, policy)

    def test_multiple_substantive_additions_are_replaced_with_stop(self) -> None:
        answer = (
            "Target: one target\nKnown baseline: Weyl\nDiscriminator: measurable prediction\n"
            "Proposal: add a new scalar field and an extra dimension.\n"
            "Red team: both are costly.\nVerdict: measurable prediction."
        )
        guarded = enforce_theorizing_output(answer, TheorizingDecision(active=True), LOCAL_AGENT_NAME)
        self.assertIn(STOP_SENTENCE, guarded)
        self.assertNotIn("extra dimension", guarded)

    def test_proposal_without_red_team_is_replaced_with_stop(self) -> None:
        answer = "Target: x\nKnown baseline: y\nDiscriminator: z\nProposal: add a new scalar.\nVerdict: new constraint."
        guarded = enforce_theorizing_output(answer, TheorizingDecision(active=True), LOCAL_AGENT_NAME)
        self.assertIn(STOP_SENTENCE, guarded)

    def test_proposal_without_replacement_check_is_stopped(self) -> None:
        answer = (
            "Target: x\nKnown baseline: y\nDiscriminator: measurable prediction\n"
            "Proposal: add a new scalar field.\nRed team: constrained.\nVerdict: measurable prediction."
        )
        guarded = enforce_theorizing_output(answer, TheorizingDecision(active=True), LOCAL_AGENT_NAME)
        self.assertIn(STOP_SENTENCE, guarded)

    def test_replacing_defining_mechanism_requires_stop(self) -> None:
        answer = (
            "Target: x\nKnown baseline: y\nDiscriminator: measurable prediction\n"
            "Proposal: remove the original mechanism.\n"
            "Replacement check: This removes the defining mechanism, so it is a new theory.\n"
            "Red team: it may be testable.\nVerdict: measurable prediction."
        )
        guarded = enforce_theorizing_output(answer, TheorizingDecision(active=True), LOCAL_AGENT_NAME)
        self.assertIn(STOP_SENTENCE, guarded)
        self.assertIn("new theory rather than a repair", guarded)

    def test_replacement_labeled_new_theory_and_stopped_is_allowed(self) -> None:
        answer = (
            "Target: x\nKnown baseline: y\nDiscriminator: measurable prediction\n"
            "Proposal: remove the original mechanism.\n"
            "Replacement check: This removes the defining mechanism, so it is a new theory.\n"
            f"Red team: the original theory is no longer being repaired.\nVerdict: {STOP_SENTENCE}"
        )
        guarded = enforce_theorizing_output(answer, TheorizingDecision(active=True), LOCAL_AGENT_NAME)
        self.assertEqual(guarded, answer)

    def test_replacement_cannot_volunteer_alternative_family(self) -> None:
        answer = (
            "Target: x\nKnown baseline: y\nDiscriminator: measurable prediction\n"
            "Proposal: remove the original mechanism.\n"
            "Replacement check: This removes the defining mechanism, so it is a new theory.\n"
            f"Red team: it is a replacement.\nVerdict: {STOP_SENTENCE} Alternatively, try string theory."
        )
        guarded = enforce_theorizing_output(answer, TheorizingDecision(active=True), LOCAL_AGENT_NAME)
        self.assertIn(STOP_SENTENCE, guarded)
        self.assertNotIn("string theory", guarded.lower())

    def test_equation_reproduction_without_gain_is_not_a_physical_solution(self) -> None:
        answer = (
            "Target: x\nKnown baseline: a known framework\nDiscriminator: none\n"
            "Proposal: use the known framework.\n"
            "Replacement check: It preserves the defining mechanism.\n"
            "Red team: the framework reproduces the equations.\n"
            "Verdict: It is a physical solution."
        )
        guarded = enforce_theorizing_output(answer, TheorizingDecision(active=True), LOCAL_AGENT_NAME)
        self.assertIn(STOP_SENTENCE, guarded)
        self.assertIn("embedding or reformulation", guarded)

    def test_reproduction_labeled_reformulation_and_stopped_is_allowed(self) -> None:
        answer = (
            "Target: x\nKnown baseline: the known framework reproduces the equations.\n"
            "Discriminator: none.\n"
            f"Verdict: This is an embedding or reformulation, not a physical solution. {STOP_SENTENCE}"
        )
        guarded = enforce_theorizing_output(answer, TheorizingDecision(active=True), LOCAL_AGENT_NAME)
        self.assertEqual(guarded, answer)

    def test_equation_reproduction_with_new_constraint_can_proceed(self) -> None:
        answer = (
            "Target: x\nKnown baseline: a known framework\nDiscriminator: new constraint\n"
            "Proposal: use the known framework.\n"
            "Replacement check: It preserves the defining mechanism.\n"
            "Red team: the framework reproduces the equations and imposes a new constraint.\n"
            "Verdict: new constraint."
        )
        guarded = enforce_theorizing_output(answer, TheorizingDecision(active=True), LOCAL_AGENT_NAME)
        self.assertEqual(guarded, answer)

    def test_designated_skeptic_cannot_make_a_competing_proposal(self) -> None:
        answer = "Red team: weak.\nProposal: add a new field.\nVerdict: no gain."
        decision = TheorizingDecision(active=True, role="skeptic")
        self.assertIn(STOP_SENTENCE, enforce_theorizing_output(answer, decision, LOCAL_AGENT_NAME))

    def test_speculative_crew_is_capped_at_proposer_and_skeptic(self) -> None:
        prompts: list[str] = []

        def transport(agent: str, from_agent: str, message: str, conversation_id: str) -> str:
            prompts.append(message)
            return f"{agent} response after {from_agent}."

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(agent_orchestrator, "_record_bus_handoff"), patch.object(
                agent_orchestrator, "update_crew_state_from_result"
            ):
                result = agent_orchestrator.run_multi_agent_dialogue(
                    topic="Can you take a crack at unifying Einstein and Maxwell?",
                    agents=["Sarah", "Astrid"],
                    rounds=4,
                    output=tmp,
                    transport=transport,
                )

        self.assertEqual(result["total_turns"], 2)
        self.assertTrue(result["conservative_theorizing"]["active"])
        self.assertIn("CONSERVATIVE_THEORIZING_ROLE: proposer", prompts[0])
        self.assertIn("CONSERVATIVE_THEORIZING_ROLE: skeptic", prompts[1])


if __name__ == "__main__":
    unittest.main()
