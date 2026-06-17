from langgraph.graph import StateGraph, START, END
from agents.state import AgentState
from agents.router_agent import classify, route
from agents.gtm_agent import gtm_retrieve, pricing_gate, route_pricing, collect_email, route_email, gtm_generate
from agents.outreach_agent import outreach_research, outreach_generate, send_gate, route_send, outreach_send
from agents.crm_agent import crm_research, crm_generate


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("classify", classify)
    g.add_node("gtm_retrieve", gtm_retrieve)
    g.add_node("pricing_gate", pricing_gate)
    g.add_node("collect_email", collect_email)
    g.add_node("gtm_generate", gtm_generate)
    g.add_node("outreach_research", outreach_research)
    g.add_node("outreach_generate", outreach_generate)
    g.add_node("send_gate", send_gate)
    g.add_node("outreach_send", outreach_send)
    g.add_node("crm_research", crm_research)
    g.add_node("crm_generate", crm_generate)

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route, {
        "gtm": "gtm_retrieve",
        "outreach": "outreach_research",
        "crm": "crm_research",
    })

    # GTM pipeline
    g.add_edge("gtm_retrieve", "pricing_gate")
    g.add_conditional_edges("pricing_gate", route_pricing, {
        "not_pricing": "gtm_generate",
        "pricing": "collect_email",
    })
    g.add_conditional_edges("collect_email", route_email, {
        "valid": "gtm_generate",
        "no_email": END,
    })
    g.add_edge("gtm_generate", END)

    # Outreach pipeline
    g.add_edge("outreach_research", "outreach_generate")
    g.add_edge("outreach_generate", "send_gate")
    g.add_conditional_edges("send_gate", route_send, {
        "review": END,
        "send": "outreach_send",
    })
    g.add_edge("outreach_send", END)

    # CRM pipeline
    g.add_edge("crm_research", "crm_generate")
    g.add_edge("crm_generate", END)

    return g.compile()


graph = build_graph()
