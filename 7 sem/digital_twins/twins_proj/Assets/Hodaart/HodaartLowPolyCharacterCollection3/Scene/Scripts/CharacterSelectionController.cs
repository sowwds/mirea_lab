using System.Collections;
using UnityEngine;


namespace LowPolyCharacterCollection3
{
    public class CharacterSelectionController : MonoBehaviour
    {
        [SerializeField] private GameObject[] characters;

        private int currentIndex = 0;
        private Animator currentAnimator;

        [SerializeField] private float switchDelay = 0.5f;
        private bool isSwitching = false;
        void Start()
        {
            ShowCharacter(currentIndex);
        }

        void ShowCharacter(int index)
        {
            for (int i = 0; i < characters.Length; i++)
            {
                characters[i].SetActive(i == index);
            }

            currentAnimator = characters[index].GetComponent<Animator>();
        }

        public void NextCharacter()
        {
            StartCoroutine(SwitchCharacterWithDelay(+1));
        }

        public void PreviousCharacter()
        {
            StartCoroutine(SwitchCharacterWithDelay(-1));
        }


        IEnumerator SwitchCharacterWithDelay(int direction)
        {
            if (isSwitching) yield break;
            isSwitching = true;

            PlayIdle();
            yield return new WaitForSeconds(switchDelay);

            currentIndex += direction;
            if (currentIndex >= characters.Length) currentIndex = 0;
            if (currentIndex < 0) currentIndex = characters.Length - 1;

            ShowCharacter(currentIndex);

            isSwitching = false;
        }
        // ====================== Animations ======================

        public void PlayWalk()
        {
            if (currentAnimator == null) return;
            PlayIdle();
            currentAnimator.SetBool("Walking", true);
        }

        public void PlayRun()
        {
            if (currentAnimator == null) return;
            PlayIdle();
            currentAnimator.SetBool("Running", true);
        }

        public void PlayJump()
        {
            if (currentAnimator == null) return;
            PlayIdle();
            currentAnimator.SetTrigger("Jump");
        }

        public void PlayDance1()
        {
            if (currentAnimator == null) return;
            PlayIdle();
            currentAnimator.SetBool("Dance 1",true);
        }

        public void PlayDance2()
        {
            if (currentAnimator == null) return;
            PlayIdle();
            currentAnimator.SetBool("Dance 2", true);
        }

        public void PlayHappyWalk()
        {
            if (currentAnimator == null) return;
            PlayIdle();
            currentAnimator.SetBool("Happy Walk", true);
        }

        public void PlayGreeting()
        {
            if (currentAnimator == null) return;
            PlayIdle();
            currentAnimator.SetTrigger("Greeting");
        }
        public void PlayIdle()
        {
            if (currentAnimator == null) return;
            currentAnimator.SetBool("Walking", false);
            currentAnimator.SetBool("Running", false);
            currentAnimator.SetBool("Dance 1", false);
            currentAnimator.SetBool("Dance 2", false);
            currentAnimator.SetBool("Happy Walk", false);
        }
    }
}
