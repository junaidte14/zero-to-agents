#Code: the full pipeline, run once, measured every way

#1 Pretraining the base model on Task A

#base_model = TinyGPT(r=0)   # no LoRA yet -- a plain, fully-trainable model
# ... trained with masked loss (Episode 05.05) on 200 reversal examples ...
#print("Base model pretraining final loss:", loss.item())
#print("Task A test accuracy, base model:", evaluate(base_model, task_A_test, make_rev_example))

#Base model pretraining final loss: 0.0000789
#Task A (reverse) test accuracy, base model: (100, 100)

#2 Freezing the base, attaching LoRA, and adapting to Task B

""" lora_model = TinyGPT(r=4)
lora_model.load_state_dict(base_model.state_dict(), strict=False)   # copy the pretrained weights in
for name, p in lora_model.named_parameters():
    p.requires_grad = ('A_q' in name or 'B_q' in name or 'A_v' in name or 'B_v' in name)   # freeze everything else

# ... trained ONLY the LoRA parameters, masked loss, on 150 sort examples ...
print("LoRA fine-tuning final loss:", loss.item())
print("Task B test accuracy, LoRA-adapted model:", evaluate(lora_model, task_B_test, make_sort_example)) """

#LoRA fine-tuning final loss: 0.000986
#Task B (sort) test accuracy, LoRA-adapted model: (83, 100)

#3 The two questions that matter most — interference and recoverability

""" lora_model.set_lora_active(True)
acc_A_attached = evaluate(lora_model, task_A_test, make_rev_example)
lora_model.set_lora_active(False)
acc_A_detached = evaluate(lora_model, task_A_test, make_rev_example)

print("Task A accuracy, LoRA ATTACHED: ", acc_A_attached)
print("Task A accuracy, LoRA DETACHED: ", acc_A_detached) """

#Task A (reverse) accuracy, LoRA ATTACHED:  (12, 100)
#Task A (reverse) accuracy, LoRA DETACHED:  (100, 100)